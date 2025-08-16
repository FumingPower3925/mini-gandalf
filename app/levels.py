import yaml
from typing import Dict, Any

def get_level(level_name: str) -> Dict[str, Any]:
    level_file: str = f"prompts/{level_name.lower().replace(' ', '_')}.yml"
    with open(level_file, "r") as f:
        level_data: Dict[str, Any] = yaml.safe_load(f)
        return level_data