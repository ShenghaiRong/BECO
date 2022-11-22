from datetime import datetime
import os

"""
All functions in this file is used to processing args passed by CLI. They should
only have `args` as parameter
"""

def init_logging_path(args):
    """
    Preparing directory path for storing training infomation
    
    If both tag and id are not given, currnet time is used for naming logging folder
    """
    # Default `logging_tag` and `run_id` are empty strings
    if len(args.logging_tag) == 0 and len(args.run_id) == 0:
        suffix = datetime.today().strftime('%y%m%d_%H%M%S')
    else:
        suffix = args.logging_tag + args.run_id
        
    args.logging_path = os.path.join(args.logging_path_base, suffix)

def init_gpus(args):
    "Setting up GPU related paramters"
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
