import yaml
import argparse
from argparse import Namespace

class YamlArgs:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.config = self._load_yaml()

    def _load_yaml(self):
        with open(self.yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def _flatten_dict(self, d, parent_key="", sep="."):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    result[sub_k] = sub_v
            else:
                result[k] = v
        return result

    def to_args(self):
        flat_config = self._flatten_dict(self.config)
        return Namespace(**flat_config)

    def add_to_parser(self, parser):
        flat_config = self._flatten_dict(self.config)
        for key, value in flat_config.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", action="store_true", default=value)
            elif isinstance(value, list):
                parser.add_argument(f"--{key}", type=type(value[0]), default=value, nargs="+")
            else:
                parser.add_argument(f"--{key}", type=type(value), default=value)
        return parser