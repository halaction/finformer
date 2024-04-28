import re
import yaml

from typing import Dict


def snake_case(string: str):
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    string = pattern.sub('_', string).lower()
    return string


class BaseConfig:

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, key):
        value = self._cfg[key]
        if isinstance(value, dict):
            return BaseConfig(value)
        return value


class FinformerConfig(BaseConfig):

    def __init__(self, cfg: Dict = None):

        if cfg is None:
            path = 'config.yaml'
            with open(path, 'r', encoding='utf-8') as file:
                cfg = yaml.safe_load(file)

        super().__init__(cfg)