from time import sleep
from typing import Dict, Any
import argparse
import logging
import platform
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from datasets.voc import VOCSegmentationPseuMask
from datasets.transforms.compose import Compose
from datasets.transforms.transform import *
from datasets import get_dataloaders

from networks.deeplabv3plus import DeepLabV3Plus

from models.train_beco import BECO

from runners.optimizer import get_optimizer
from runners.scheduler import get_scheduler
from runners.epoch_runner import EpochRunner

import utils
from utils.distributed import is_distributed, get_dist_info, get_device


# set global level to WARNING
logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)
# disable global warning
warnings.filterwarnings("ignore")


def get_args() -> Dict[str, Any]:
    """
    All runtime specific configs should be written here to avoid modifying 
    config file
    """
    parser = argparse.ArgumentParser()

    # External config file
    #parser.add_argument("--config", type=str, default=None,
    #                    help="path to config file")

    # Distributed and Performance Options
    parser.add_argument("--amp", action='store_true', default=False,
                        help="use automatic mixed precision if set to true")
    parser.add_argument('-dist', '--distributed', action='store_true', 
                        default=False,
                        help='Use multi-processing distributed training to '
                             'launch N processes per node, which has N GPUs.')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Use deterministic algorithm, '
                             'may reduce performance')
    parser.add_argument('--gpu_id', type=str, default=None,
                        help="Optional, specify gpu(s) to use, leave it empty \
                            to use all gpus, example: 0,1,2,3")

    # Misc
    parser.add_argument("--run_id", type=str, default="",
                        help="Optional, add this to logging folder name")
    parser.add_argument("--logging_path_base", type=str, 
                        default="./data/logging",
                        help="Path to write logging and checkpoint")
    parser.add_argument("--logging_tag", type=str, default="",
                        help="Optional, add an extra tag to \
                            logging folder name")
    parser.add_argument("--seed", type=int, default=3223,
                        help="random seed")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="Path to checkpoint to restore, will resume from \
                            given ckpt if this is a valid checkpoint")
    parser.add_argument("--test", action='store_true', default=False,
                        help="Run in test mode")

    return parser.parse_args()


def main():
    # get CLI params and pre-processing
    args = get_args()
    args_preprocess(args)


    # launtch DDP
    if args.distributed:
        utils.distributed.init_dist()
        #mp.set_sharing_strategy('file_system')
        mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(args, ))
    else:
        main_worker(None, args)
    

def main_worker(gpu, args: Dict[str, Any]):
    # init DDP
    if args.distributed:
        dist.init_process_group(
            backend="gloo" if platform.system() == "Windows" else 'nccl',
            rank=gpu,
            world_size=torch.cuda.device_count()
        )

    # init logger
    logger = utils.logger.R0Logger(__name__)
    log_dist_info(logger, args)

    # create work dir
    utils.files.makedir(args.logging_path)
    logger.info(f"work dir created at {args.logging_path}")

    # build config from config file
    #logger.info("Building configs...")
    #cfg = Config.build_config(args.config)
    #replace_vars(cfg, args)
    #Config.dump_config(cfg, args.logging_path, gpu)

    # setup seed
    utils.misc.set_seed(args.seed)

    # setup torch cuda options
    utils.misc.set_cuda(args)

    # setup datasets
    logger.info("Building datasets...")

    train_transform = Compose([
        RandomResizedCropMask(size=[512, 512], scale=[0.5, 0.75, 1, 1.5, 1.75, 2.0]),
        RandomHorizontalFlipMask(),
        ToTensorMask(),
        NormalizeMask(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize(size=[512, 512]),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VOCSegmentationPseuMask(
                                            root='./data/VOC2012',
                                            mode='train',
                                            transform=train_transform,
                                            is_aug=True, 
                                            mask_dir='./data/irn_pseudo_label',
                                            pseumask_dir='./data/irn_mask')
    val_dataset = VOCSegmentationPseuMask(
                                            root='./data/VOC2012',
                                            mode='val',
                                            transform=val_transform)
    test_dataset = VOCSegmentationPseuMask(
                                            root='./data/VOC2012',
                                            mode='val',
                                            transform=val_transform)

    datasets_dict = dict(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset
    )

    dataloader_dict, sampler_dict = get_dataloaders(
        config={
			"batch_size": 8, #per GPU
			"num_workers": 1,
			"drop_last": True
		},
        datasets_dict=datasets_dict,
        seed=args.seed
    )



    # setup network
    logger.info("Building networks...")

    network1 = DeepLabV3Plus(
        backbone={
			"pretrain": "./data/model_zoo/resnetv1d101_mmcv.pth",
			"variety": "resnet-D",
			"depth": 101,
			"out_indices": [1, 4],
			"output_stride": 16,
			"contract_dilation": False,
			"multi_grid": True,
			"norm_layer": "SyncBatchNorm"
        },
        decoder={
            "type": "SepASPP",
			"in_channels": 2048,
			"channels": 256,
			"lowlevel_in_channels": 256,
			"lowlevel_channels": 48,
			"atrous_rates": [6, 12, 18],
			"dropout_ratio": 0.1,
			"num_classes": 21,
			"norm_layer": "SyncBatchNorm",
			"align_corners": False
        }
    )
    network1 = network1.to(get_device())

    network2 = DeepLabV3Plus(
        backbone={
			"pretrain": "./data/model_zoo/resnetv1d101_mmcv.pth",
			"variety": "resnet-D",
			"depth": 101,
			"out_indices": [1, 4],
			"output_stride": 16,
			"contract_dilation": False,
			"multi_grid": True,
			"norm_layer": "SyncBatchNorm"
        },
        decoder={
            "type": "SepASPP",
			"in_channels": 2048,
			"channels": 256,
			"lowlevel_in_channels": 256,
			"lowlevel_channels": 48,
			"atrous_rates": [6, 12, 18],
			"dropout_ratio": 0.1,
			"num_classes": 21,
			"norm_layer": "SyncBatchNorm",
			"align_corners": False
        }
    )
    network2 = network2.to(get_device())

    nets_dict = dict(
        network1=network1,
        network2=network2
    )

    # setup model
    logger.info("Building model...")

    beco_config = {
            "ignore_bg": True,
			"mix_aug": True,
			"mix_prob": 0.5,
			"bdry_size": 1,
			"bdry_whb": 0.2,
			"bdry_whi": 0.2,
			"bdry_wlb": 0,
			"bdry_wli": 0,
			"warm_up": 1,
			"highres_t": 0.95,
			"save_logits": True,
			"test_msc": False
        }
    model = BECO(
        **beco_config,
        logging_path=args.logging_path,
        config={'datasets': {"num_classes": 21}},
        nets_dict=nets_dict
    )

    model.set_up_amp(args.amp)

    # setup runner
    logger.info("Building runner...")
    #runner = runners.build_runner(args, cfg, model, datasets_dict)
    _net_dict = utils.distributed.skip_module_for_dict(model.nets)
    optim = get_optimizer(
        optim_config={
            "type": "SGD",
            "settings":{
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 1e-4
            }
        },
        net_configs=[{"name": "network1"}, {"name": "network2"}],
        net_dict=_net_dict
    )

    sched = get_scheduler(
        sched_configs={
            "type": "PolyLR",
            "settings": {
                "lr_power": 0.9,
                "min_lr": 1e-5
            }
        },
        optim=optim
    )

    runner = EpochRunner(args, model, optim, sched, dataloader_dict, sampler_dict,
                         **{
                            "max_epoch": 80,
                            "workflow": {
                                "train": 1,
                                "val": 1
                                }
                            }
                        )


    # To prevent OOM under some circumstances (e.g., required mem nears the maximum mem)
    # Though it is not needed in most circumstances
    torch.cuda.empty_cache()

    # start training
    if not args.test:
        runner.run()
        # Wait for a while before test to prevent some IO error
        sleep(2)

    # start testing
    runner.test()

    # exiting, doing cleaning
    runner.close()


# -------------------------
# Aux functions
def args_preprocess(args):
    utils.argparser.init_logging_path(args)
    utils.argparser.init_gpus(args)

def log_dist_info(logger, args):
    if is_distributed():
        logger.info(f"Use {get_dist_info()[1]} GPUs for distributed training")
    else:
        logger.info("Distributed training is disabled")

    if args.amp:
        logger.info("AMP training is enabled")
    else:
        logger.info("AMP training is disabled")



# Entrance
if __name__ == "__main__":
    main()
