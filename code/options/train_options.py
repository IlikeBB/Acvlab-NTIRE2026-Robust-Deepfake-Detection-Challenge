from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=5)
        parser.add_argument('--data_aug', action='store_true', default=False, help='perform additional data augmentation')
        parser.add_argument('--augment', dest='data_aug', action='store_true', default=False, help='alias for --data_aug')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [adam, adamw, sgd]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--last_epoch', type=int, default=-1)
        parser.add_argument('--train_split', type=str, default='train')
        parser.add_argument('--val_split', type=str, default='val')
        parser.add_argument('--niter', type=int, default=50)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--residual_lr', type=float, default=None)
        parser.add_argument('--grad_clip', type=float, default=1.0)
        parser.add_argument('--use_amp', '--amp', dest='use_amp', action='store_true', default=False)
        parser.add_argument('--clear_cache_each_epoch', action='store_true', default=True)
        parser.add_argument('--no_clear_cache_each_epoch', action='store_true')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--deterministic', action='store_true', default=True)
        parser.add_argument('--no_deterministic', action='store_true')
        parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none', 'cosine'])
        parser.add_argument('--cosine_t_max', type=int, default=None)
        parser.add_argument('--cosine_min_lr', type=float, default=1e-6)
        parser.add_argument('--lr_warmup_steps', type=float, default=0.0)
        self.isTrain = True
        return parser
