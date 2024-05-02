import re
import yaml
import json

from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import torch


def adaptive_get(url):

    session = requests.session() 
    retries = Retry(
        total = 5,
        backoff_factor = 1,
        status_forcelist = [429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)

    response = session.get(url)
    
    return response


def is_not_none(x):
        return x is not None
    

def filter_none(array):

    filtered_array = list(filter(is_not_none, array))

    return filtered_array


def snake_case(string: str):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    string = pattern.sub('_', string).lower()
    return string


def get_device():

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    return device


class DotDict(dict):
    """
    A dictionary that supports dot access notation
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):

        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)


class BaseConfig:

    def __init__(self, cfg):
        self._cfg = cfg

    def __getitem__(self, key):
        
        value = self._cfg[key]
        if isinstance(value, dict):
            return BaseConfig(value)

        return value
    
    def __setitem__(self, key, value):
        self._cfg[key] = value

    def __getattr__(self, key):
        if hasattr(dict, key):
            return getattr(self._cfg, key)
        else:
            value = self._cfg[key]
            if isinstance(value, dict):
                return BaseConfig(value)
        return value
    
    def __repr__(self):
        return json.dumps(self._cfg, indent=2)
    

class FinformerConfig(DotDict):

    def __init__(self, cfg: Dict = None):

        if cfg is None:
            path = 'old_config.yaml'
            with open(path, 'r', encoding='utf-8') as file:
                cfg = yaml.safe_load(file)

        super().__init__(cfg)

    def __repr__(self):
        return json.dumps(self, indent=2)


class FinformerBatch(DotDict):

    def __init__(
        self, 
        **kwargs,
    ):
        
        batch = dict()
        for key, value in kwargs.items():
            batch[key] = value

        super().__init__(batch)

        

        