import yaml
from typing import Dict, Any, Optional, Union


class BaseModel:
    require_system_prompt: bool

    def load_config(self, config: Optional[Union[str, dict]]) -> Dict:
        if isinstance(config, dict):
            return config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        return config

    def process_fn(self):
        raise NotImplementedError

    def generate(self, data: Any):
        raise NotImplementedError
