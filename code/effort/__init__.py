from .svd_residual import apply_svd_residual_to_self_attn, SVDResidualLinear


def __getattr__(name):
    if name in {"EffortCLIP", "get_model"}:
        from .model import EffortCLIP, get_model
        globals()["EffortCLIP"] = EffortCLIP
        globals()["get_model"] = get_model
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EffortCLIP",
    "get_model",
    "apply_svd_residual_to_self_attn",
    "SVDResidualLinear",
]
