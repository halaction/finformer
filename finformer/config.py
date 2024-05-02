from typing import List

from hydra import compose, initialize
from omegaconf import OmegaConf


def get_config(overrides: List[str] = None):

    with initialize(version_base=None, config_path='config', job_name='finformer'):
        config = compose(config_name='config', overrides=overrides)

    print(OmegaConf.to_yaml(config))

    return config
