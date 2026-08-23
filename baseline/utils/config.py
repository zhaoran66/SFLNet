import yaml
from typing import Dict


class ConfigDict:
    def __init__(self, data: Dict):
        for k, v in data.items():
            if isinstance(v, dict):
                self.__dict__[k] = ConfigDict(v)
            else:
                self.__dict__[k] = v
    
    def __getattr__(self, name):
        return None


def load_config(config_path: str) -> ConfigDict:
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return ConfigDict(config_data)
