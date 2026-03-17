import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # Original weights (fixed)
        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        # Buffers for regularizers (must follow device + be in state_dict)
        # English: register_buffer ensures tensors move with .to(device) and are saved in checkpoints.
        self.register_buffer("weight_original_fnorm", torch.tensor(0.0))
        self.register_buffer("weight_main_fnorm", torch.tensor(0.0))
        # Single residual expert (legacy path)
        self.S_residual = None
        self.U_residual = None
        self.V_residual = None
        self.S_r = None
        self.U_r = None
        self.V_r = None

        # Optional multi-expert residual bank
        self.num_experts = 1
        self.S_residual_extra = nn.ParameterList()
        self.U_residual_extra = nn.ParameterList()
        self.V_residual_extra = nn.ParameterList()
        self._route_alpha = None
        # Residual fusion mode:
        # - "add": legacy behavior
        # - "gated": base + sigmoid(gate_logit) * residual
        # - "softplus": base + softplus(gate_logit) * residual
        self.residual_mode = "add"
        self.register_parameter("residual_gate_logit", None)
        # Which SVD residual factors are trainable.
        # - "full": train U/S/V
        # - "sigma_only": train only S (spectral reweighting)
        # - "uv_only": train only U/V (basis update with fixed spectrum)
        self.residual_train_mode = "full"

    def _has_primary_residual(self):
        return (
            (self.S_residual is not None)
            and (self.U_residual is not None)
            and (self.V_residual is not None)
        )

    @staticmethod
    def _residual_weight(U, S, V):
        return U @ torch.diag(S) @ V

    def iter_residual_params(self):
        if self._has_primary_residual():
            yield self.S_residual
            yield self.U_residual
            yield self.V_residual
        for plist in (self.S_residual_extra, self.U_residual_extra, self.V_residual_extra):
            for p in plist:
                yield p

    def set_num_experts(self, num_experts=1, init="copy"):
        target = max(1, int(num_experts or 1))

        if not self._has_primary_residual():
            self.num_experts = 1
            self.S_residual_extra = nn.ParameterList()
            self.U_residual_extra = nn.ParameterList()
            self.V_residual_extra = nn.ParameterList()
            return

        if target <= 1:
            self.num_experts = 1
            self.S_residual_extra = nn.ParameterList()
            self.U_residual_extra = nn.ParameterList()
            self.V_residual_extra = nn.ParameterList()
            return

        cur = 1 + len(self.S_residual_extra)
        if cur < target:
            for _ in range(target - cur):
                if str(init).lower() == "zero":
                    s = torch.zeros_like(self.S_residual.data)
                    u = torch.zeros_like(self.U_residual.data)
                    v = torch.zeros_like(self.V_residual.data)
                else:
                    s = self.S_residual.data.clone()
                    u = self.U_residual.data.clone()
                    v = self.V_residual.data.clone()
                self.S_residual_extra.append(nn.Parameter(s))
                self.U_residual_extra.append(nn.Parameter(u))
                self.V_residual_extra.append(nn.Parameter(v))
        elif cur > target:
            keep = target - 1
            self.S_residual_extra = nn.ParameterList(list(self.S_residual_extra)[:keep])
            self.U_residual_extra = nn.ParameterList(list(self.U_residual_extra)[:keep])
            self.V_residual_extra = nn.ParameterList(list(self.V_residual_extra)[:keep])

        self.num_experts = target

    def set_route_alpha(self, alpha):
        self._route_alpha = alpha

    def clear_route_alpha(self):
        self._route_alpha = None

    def set_residual_mode(self, mode="add", gate_logit_init=0.0):
        m = str(mode or "add").lower()
        if m not in {"add", "gated", "softplus"}:
            m = "add"
        self.residual_mode = m
        if m in {"gated", "softplus"}:
            if self.residual_gate_logit is None:
                self.residual_gate_logit = nn.Parameter(torch.tensor(float(gate_logit_init)), requires_grad=True)
            else:
                with torch.no_grad():
                    self.residual_gate_logit.fill_(float(gate_logit_init))

    def set_residual_train_mode(self, mode="full"):
        m = str(mode or "full").lower()
        if m not in {"full", "sigma_only", "uv_only"}:
            m = "full"
        self.residual_train_mode = m

        for p in self.iter_residual_params():
            p.requires_grad = False

        triplets = self._expert_triplets()
        for Ue, Se, Ve in triplets:
            if m == "full":
                Ue.requires_grad = True
                Se.requires_grad = True
                Ve.requires_grad = True
            elif m == "sigma_only":
                Se.requires_grad = True
            elif m == "uv_only":
                Ue.requires_grad = True
                Ve.requires_grad = True

    def _residual_scale(self, x: torch.Tensor):
        if self.residual_gate_logit is None:
            return x.new_tensor(1.0)
        logit = self.residual_gate_logit.to(dtype=x.dtype, device=x.device)
        if self.residual_mode == "gated":
            return torch.sigmoid(logit)
        if self.residual_mode == "softplus":
            return F.softplus(logit)
        return x.new_tensor(1.0)

    def _expert_triplets(self):
        if not self._has_primary_residual():
            return []

        out = [(self.U_residual, self.S_residual, self.V_residual)]
        n_extra = min(
            int(self.num_experts) - 1,
            len(self.U_residual_extra),
            len(self.S_residual_extra),
            len(self.V_residual_extra),
        )
        for i in range(max(0, n_extra)):
            out.append((self.U_residual_extra[i], self.S_residual_extra[i], self.V_residual_extra[i]))
        return out

    def _normalized_alpha(self, x, k):
        alpha = self._route_alpha
        if alpha is None:
            return x.new_full((k,), 1.0 / float(k))

        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=x.device, dtype=x.dtype)
        else:
            alpha = alpha.to(device=x.device, dtype=x.dtype)

        if alpha.dim() <= 0:
            alpha = alpha.view(1)

        if alpha.dim() == 1:
            if alpha.numel() < k:
                pad = torch.zeros(k - alpha.numel(), device=alpha.device, dtype=alpha.dtype)
                alpha = torch.cat([alpha, pad], dim=0)
            elif alpha.numel() > k:
                alpha = alpha[:k]
            alpha = alpha.clamp_min(0.0)
            alpha = alpha / alpha.sum().clamp_min(1e-12)
            return alpha

        if alpha.dim() == 2:
            b = int(x.shape[0]) if x.dim() > 0 else int(alpha.shape[0])
            if alpha.shape[0] != b:
                if alpha.shape[0] == 1:
                    alpha = alpha.expand(b, -1)
                else:
                    alpha = alpha[:1].expand(b, -1)
            if alpha.shape[1] < k:
                pad = torch.zeros(alpha.shape[0], k - alpha.shape[1], device=alpha.device, dtype=alpha.dtype)
                alpha = torch.cat([alpha, pad], dim=1)
            elif alpha.shape[1] > k:
                alpha = alpha[:, :k]
            alpha = alpha.clamp_min(0.0)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return alpha

        return x.new_full((k,), 1.0 / float(k))

    def compute_current_weight(self, expert_idx=0):
        triplets = self._expert_triplets()
        if not triplets:
            return self.weight_main
        idx = max(0, min(int(expert_idx), len(triplets) - 1))
        U, S, V = triplets[idx]
        return self.weight_main + self._residual_weight(U, S, V)

    def forward(self, x):
        triplets = self._expert_triplets()
        if not triplets:
            return F.linear(x, self.weight_main, self.bias)

        # Legacy single-expert path keeps exact old behavior.
        if len(triplets) == 1:
            U, S, V = triplets[0]
            residual_weight = self._residual_weight(U, S, V)
            if self.residual_mode in {"gated", "softplus"}:
                g = self._residual_scale(x)
                return F.linear(x, self.weight_main + g * residual_weight, self.bias)
            return F.linear(x, self.weight_main + residual_weight, self.bias)

        base = F.linear(x, self.weight_main, self.bias)
        residual_outputs = []
        for U, S, V in triplets:
            residual_outputs.append(F.linear(x, self._residual_weight(U, S, V), None))
        stack = torch.stack(residual_outputs, dim=-1)

        k = len(triplets)
        alpha = self._normalized_alpha(x, k)
        if alpha.dim() == 1:
            view_shape = [1] * (stack.dim() - 1) + [k]
            mix = (stack * alpha.view(*view_shape)).sum(dim=-1)
        else:
            view_shape = [stack.shape[0]] + [1] * max(0, stack.dim() - 2) + [k]
            mix = (stack * alpha.view(*view_shape)).sum(dim=-1)

        if self.residual_mode in {"gated", "softplus"}:
            g = self._residual_scale(x)
            return base + g * mix
        return base + mix

    def compute_orthogonal_loss(self):
        triplets = self._expert_triplets()
        if (not triplets) or (self.U_r is None) or (self.V_r is None):
            return 0.0

        loss = 0.0
        for Ue, _, Ve in triplets:
            UUT = torch.cat((self.U_r, Ue), dim=1) @ torch.cat((self.U_r, Ue), dim=1).t()
            VVT = torch.cat((self.V_r, Ve), dim=0) @ torch.cat((self.V_r, Ve), dim=0).t()

            UUT_identity = torch.eye(UUT.size(0), device=UUT.device)
            VVT_identity = torch.eye(VVT.size(0), device=VVT.device)
            loss = loss + 0.5 * torch.norm(UUT - UUT_identity, p="fro")
            loss = loss + 0.5 * torch.norm(VVT - VVT_identity, p="fro")

        return loss / float(len(triplets))

    def compute_keepsv_loss(self):
        triplets = self._expert_triplets()
        if not triplets:
            return 0.0

        loss = 0.0
        for Ue, Se, Ve in triplets:
            weight_current = self.weight_main + self._residual_weight(Ue, Se, Ve)
            weight_current_fnorm = torch.norm(weight_current, p="fro")
            loss = loss + torch.abs(weight_current_fnorm ** 2 - self.weight_original_fnorm ** 2)
        return loss / float(len(triplets))

    def compute_fn_loss(self):
        triplets = self._expert_triplets()
        if not triplets:
            return 0.0

        loss = 0.0
        for Ue, Se, Ve in triplets:
            weight_current = self.weight_main + self._residual_weight(Ue, Se, Ve)
            weight_current_fnorm = torch.norm(weight_current, p="fro")
            loss = loss + (weight_current_fnorm ** 2)
        return loss / float(len(triplets))

    def compute_expert_div_loss(self):
        triplets = self._expert_triplets()
        if len(triplets) <= 1:
            return 0.0

        vecs = []
        for Ue, Se, Ve in triplets:
            w = self._residual_weight(Ue, Se, Ve).reshape(-1)
            w = w / w.norm().clamp_min(1e-12)
            vecs.append(w)

        loss = 0.0
        pairs = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                loss = loss + (vecs[i] * vecs[j]).sum().abs()
                pairs += 1
        return loss / float(max(1, pairs))


def apply_svd_residual_to_self_attn(model, r):
    return apply_svd_residual_to_self_attn_ex(model, r, svd_device=None, show_progress=False, fast_init=False)


def _is_supported_attn_linear(name: str) -> bool:
    if "self_attn" in name:
        return True
    # timm EVA/ViT attention naming
    if ".attn." in name:
        leaf = name.split(".")[-1]
        if leaf in {"q_proj", "k_proj", "v_proj", "proj", "out_proj"}:
            return True
    return False


def _extract_block_index(name: str):
    parts = str(name).split(".")
    for i, p in enumerate(parts):
        if p in {"layers", "blocks"} and (i + 1) < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                return None
    return None


def _iter_self_attn_linear_names(model, adapter_last_n_layers=None):
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and _is_supported_attn_linear(name)):
            continue
        yield name


def _get_parent_and_attr(root, full_name):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _filter_names_by_last_n_blocks(names, last_n):
    if last_n is None:
        return names
    try:
        n = int(last_n)
    except Exception:
        return names
    if n <= 0:
        return names
    idxs = []
    for nm in names:
        bi = _extract_block_index(nm)
        if bi is not None:
            idxs.append(bi)
    if not idxs:
        return names
    keep_min = max(idxs) - n + 1
    out = []
    for nm in names:
        bi = _extract_block_index(nm)
        if bi is None or bi >= keep_min:
            out.append(nm)
    return out


def apply_svd_residual_to_self_attn_ex(
    model,
    r,
    svd_device=None,
    show_progress=True,
    fast_init=False,
    adapter_last_n_layers=None,
):
    names = list(_iter_self_attn_linear_names(model))
    names = _filter_names_by_last_n_blocks(names, adapter_last_n_layers)
    iterator = names
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(names, desc="svd-init", leave=False, mininterval=1.0)
        except Exception:
            iterator = names

    for name in iterator:
        parent, attr = _get_parent_and_attr(model, name)
        sub_module = getattr(parent, attr)
        if not isinstance(sub_module, nn.Linear):
            continue
        if fast_init:
            new_module = replace_with_svd_residual_fast(sub_module, r)
        else:
            new_module = replace_with_svd_residual(sub_module, r, svd_device=svd_device)
        setattr(parent, attr, new_module)

    for param_name, param in model.named_parameters():
        if any(x in param_name for x in ["S_residual", "U_residual", "V_residual"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def set_svd_residual_num_experts(model, num_experts=1, init="copy"):
    for m in model.modules():
        if isinstance(m, SVDResidualLinear):
            m.set_num_experts(num_experts=num_experts, init=init)
    return model


def set_svd_residual_mode(model, mode="add", gate_logit_init=0.0):
    for m in model.modules():
        if isinstance(m, SVDResidualLinear):
            m.set_residual_mode(mode=mode, gate_logit_init=gate_logit_init)
    return model


def set_svd_residual_train_mode(model, mode="full"):
    for m in model.modules():
        if isinstance(m, SVDResidualLinear):
            m.set_residual_train_mode(mode=mode)
    return model


def replace_with_svd_residual(module, r, svd_device=None):
    if not isinstance(module, nn.Linear):
        return module

    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None

    new_module = SVDResidualLinear(in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone())

    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    new_module.weight_original_fnorm.copy_(torch.norm(module.weight.data, p="fro").detach())

    w = module.weight.data
    target_device = w.device
    target_dtype = w.dtype
    if svd_device is not None:
        try:
            svd_device = torch.device(str(svd_device))
        except Exception:
            svd_device = target_device
    else:
        svd_device = target_device
    mat = w.to(device=svd_device)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    U = U.to(device=target_device, dtype=target_dtype)
    S = S.to(device=target_device, dtype=target_dtype)
    Vh = Vh.to(device=target_device, dtype=target_dtype)
    r = min(r, len(S))

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    weight_main = U_r @ torch.diag(S_r) @ Vh_r
    new_module.weight_main_fnorm.copy_(torch.norm(weight_main.data, p="fro").detach())
    new_module.weight_main.data.copy_(weight_main)

    U_residual = U[:, r:]
    S_residual = S[r:]
    Vh_residual = Vh[r:, :]

    if len(S_residual) > 0:
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())

        new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
        new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None

        new_module.S_r = None
        new_module.U_r = None
        new_module.V_r = None

    return new_module


def replace_with_svd_residual_fast(module, r):
    """Build SVDResidualLinear with correct tensor shapes without running SVD.

    Intended for fast structure materialization before loading cached state_dict.
    """
    if not isinstance(module, nn.Linear):
        return module

    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None
    k = min(out_features, in_features)
    rr = min(int(r), int(k))
    rem = int(k - rr)

    new_module = SVDResidualLinear(in_features, out_features, rr, bias=bias, init_weight=module.weight.data.clone())
    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    w = module.weight.data
    dev = w.device
    dt = w.dtype
    new_module.weight_original_fnorm.copy_(torch.norm(w, p="fro").detach())
    new_module.weight_main_fnorm.copy_(torch.norm(w, p="fro").detach())
    new_module.weight_main.data.copy_(w)

    if rem > 0:
        new_module.S_residual = nn.Parameter(torch.zeros(rem, device=dev, dtype=dt))
        new_module.U_residual = nn.Parameter(torch.zeros(out_features, rem, device=dev, dtype=dt))
        new_module.V_residual = nn.Parameter(torch.zeros(rem, in_features, device=dev, dtype=dt))
        new_module.S_r = nn.Parameter(torch.zeros(rr, device=dev, dtype=dt), requires_grad=False)
        new_module.U_r = nn.Parameter(torch.zeros(out_features, rr, device=dev, dtype=dt), requires_grad=False)
        new_module.V_r = nn.Parameter(torch.zeros(rr, in_features, device=dev, dtype=dt), requires_grad=False)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
        new_module.S_r = None
        new_module.U_r = None
        new_module.V_r = None
    return new_module
