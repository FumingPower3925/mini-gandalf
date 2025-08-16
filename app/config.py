import toml
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    with open("config.toml", "r") as f:
        return toml.load(f)