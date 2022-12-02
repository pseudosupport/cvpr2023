from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniimagenet':
        from models.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        from models.dataloader.cub import CUB as Dataset
    elif args.dataset == 'tieredimagenet':
        from models.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cifar_fs':
        from models.dataloader.cifar_fs import DatasetLoader as Dataset
    elif args.dataset == 'rareairplane':
        from models.dataloader.rareairplane import MiniImageNet as Dataset
    elif args.dataset == 'mar20':
        from models.dataloader.mar20 import MiniImageNet as Dataset
    elif args.dataset == 'hrsc':
        from models.dataloader.hrsc import MiniImageNet as Dataset
    elif args.dataset == 'mtarsi_2':
        from models.dataloader.mtarsi_2 import MiniImageNet as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
