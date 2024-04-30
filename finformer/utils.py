import re
import yaml
import json

from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from itertools import chain
import torch
import torch.nn.functional as F


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


def snake_case(string: str):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    string = pattern.sub('_', string).lower()
    return string


def right_zero_pad(tensor, max_length, dim=-1):
    
    pad = [(0, 0) for _ in range(tensor.dim())]
    pad[dim] = (0, max_length - tensor.size(dim))
    pad = tuple(chain.from_iterable(pad[::-1]))

    padded_tensor = F.pad(tensor, pad, 'constant', 0)

    return padded_tensor


def collate_dict(dict_list, pad=False):

    keys = None

    # Convert array of dicts to dict of array
    for dict_item in dict_list:
        if dict_item is not None:
            if keys is None:
                keys = dict_item.keys()
                output = {key: list() for key in keys}

            for key in keys:
                value = dict_item[key]
                output[key].append(value)

    # Concatenate arrays in dict
    for key, values in output.items():
        if pad:
            dims = [value.dim() for value in values]
            min_dim = min(dims)
            if min_dim > 1:
                lengths = [value.size(1) for value in values]
                max_length = max(lengths)
                    
                values = [right_zero_pad(value, max_length, dim=1) for value in values]

        output[key] = torch.cat(values, dim=0)

    return output


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


class FinformerConfig(BaseConfig):

    def __init__(self, cfg: Dict = None):

        if cfg is None:
            path = 'config.yaml'
            with open(path, 'r', encoding='utf-8') as file:
                cfg = yaml.safe_load(file)

        super().__init__(cfg)