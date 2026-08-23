import yaml
import argparse
from typing import Dict, Any


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)
    
    def __repr__(self) -> str:
        return str(self._config)


def load_config(config_path: str = 'config.yaml') -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SPL: Latent Space Keypoint Estimation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    return parser.parse_args()
