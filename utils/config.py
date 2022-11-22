import os
import json
from typing import Any
import yaml
import time
import warnings

from addict import Dict

import utils
from . import files


BASE_KEY = '_base_'

class Config:
    """A class holding functions to processing config files"""
    @staticmethod
    def _json_handler(filename: str) -> Dict:
        with open(filename) as f:
            cfg_dict = json.load(f)
        return cfg_dict

    @staticmethod
    def _yaml_handler(filename: str) -> Dict:
        with open(filename) as f:
            cfg_dict = yaml.load(f)
        return cfg_dict

    @staticmethod
    def _read_cfg(filename: str) -> Dict:
        if filename.endswith('.json'):
            cfg_dict = Config._json_handler(filename)
        elif filename.endswith('.yaml'):
            cfg_dict = Config._yaml_handler(filename)

        cfg_dict = Dict(cfg_dict)
        return cfg_dict

    @staticmethod
    def substitute_vars(config, var_name: str, value: Any):
        if isinstance(config, dict):
            for k, v in config.items():
                if isinstance(v, dict) or isinstance(v, list):
                    Config.substitute_vars(v, var_name, value)
                else:
                    if v == var_name:
                        config[k] = value
        elif isinstance(config, list):
            for item in config:
                Config.substitute_vars(item, var_name, value)

    @staticmethod
    def merge_dict(a: Dict, b: Dict, replace: bool=True):
        """
        merge dict `a` into `b`, if the key overlapped, set replace = True to
        use the key in `a` otherwise use the key in `b`

        NOTE: if two object lists are merged, should place an empty object 
            in `b` if the coresponding object does not need to be changed.
            Otherwiseto unwanted behavior will occur
        """
        logger = utils.logger.R0Logger(__name__)

        for k, v in a.items():
            # Merge dict
            if isinstance(v, dict) and k in b and b[k] is not None:
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f"Error occured when trying to merge"
                        f"{type(b[k])} with {type(v)}"
                        )
                else:
                    Config.merge_dict(v, b[k], replace)
            # Merge list
            elif isinstance(v, list) and k in b and b[k] is not None:
                # Target and source should both be list
                if not isinstance(b[k], list):
                    raise TypeError(
                        f"Error occured when trying to merge"
                        f"{type(b[k])} with {type(v)}"
                        )
                # Make sure v is not empty list
                # If the elements of List are dicts
                elif len(v) and isinstance(v[0], dict):
                    # When they are equal in length, merge each item
                    if len(v) == len(b[k]):
                        for i, _ in enumerate(v):
                            Config.merge_dict(v[i], b[k][i], replace)
                    # Merge base config and current config
                    else:
                        #logger.warning("Detected source and target have list of\
                        #    dict but with different length,\
                        #    the contant from source will be discarded.")
                        b[k] = v + b[k]
                # Ignore base config
                else:
                    # Usually base is merged into current config, so the item of
                    # current config is appended after base
                    #b[k] = v + b[k]
                    pass
            # Direct copy
            else:
                if k in b:
                    b[k] = v if replace else b[k]
                else:
                    b[k] = v
        return b

    @staticmethod
    def pretty_text(a: Dict):
        return json.dumps(a, indent=4, sort_keys=False)

    @staticmethod
    def build_config(filename: str):
        files.check_file_exist(filename)
        files.check_file_type(filename, [".json", ".yaml", ".py"])
        cfg_dict = Config._read_cfg(filename)
        
        if BASE_KEY in cfg_dict:
            base_list = cfg_dict.pop(BASE_KEY)
            assert isinstance(base_list, list)
            for base_cfg in base_list:
                base_cfg_dict = Config.build_config(base_cfg)
                # Merge base into current
                Config.merge_dict(base_cfg_dict, cfg_dict, replace=False)

        return cfg_dict

    @staticmethod
    def dump_config(config, path: str, gpu: int):
        """Dump current config to a file for reproduction or debugging"""
        if gpu == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            dump_file = os.path.join(path, f'config_{timestamp}.json')
            with open(dump_file, "w") as f:
                f.write(Config.pretty_text(config))

