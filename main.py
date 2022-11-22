from time import sleep
from typing import Dict, Any
import argparse
import logging
import platform
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import utils
import datasets
import models
import networks
import runners
from utils.config import Config
from utils.distributed import is_distributed, get_dist_info


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
    parser.add_argument("--config", type=str, default=None,
                        help="path to config file")

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
    logger.info("Building configs...")
    cfg = Config.build_config(args.config)
    replace_vars(cfg, args)
    Config.dump_config(cfg, args.logging_path, gpu)

    # setup seed
    utils.misc.set_seed(cfg.misc.seed)

    # setup torch cuda options
    utils.misc.set_cuda(args)

    # setup datasets
    logger.info("Building datasets...")
    datasets_dict = datasets.get_datasets(cfg.datasets, cfg.method)

    # setup network
    logger.info("Building networks...")
    nets_dict = networks.get_network(cfg.networks)

    # setup model
    logger.info("Building model...")
    model = models.get_model(cfg, nets_dict)
    model.set_up_amp(args.amp)

    # setup runner
    logger.info("Building runner...")
    runner = runners.build_runner(args, cfg, model, datasets_dict)

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

def replace_vars(cfg, args):
    Config.substitute_vars(cfg.hooks, "$LOGGING_PATH", args.logging_path)
    Config.substitute_vars(cfg.hooks, "$PROJECT", cfg.method)
    Config.substitute_vars(cfg.hooks, "$WANDB_PROJECT", 
                           cfg.get('wandb_method', cfg.method))
    Config.substitute_vars(cfg.networks, "$NORM_OP", 
                        "SyncBatchNorm" if args.distributed else "BatchNorm2d")
    Config.substitute_vars(cfg.hooks, "$OPTIM", 
                            "AMPOptimizerHook" if args.amp else "OptimizerHook")
    Config.substitute_vars(cfg.hooks, "$CKPT_PATH", args.ckpt)



# Entrance
if __name__ == "__main__":
    main()
