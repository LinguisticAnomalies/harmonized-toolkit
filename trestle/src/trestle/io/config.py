from configparser import ConfigParser
from pathlib import Path

def load_config(config_path: str | Path):
    config_path = Path(config_path)

    # redefine the path
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = ConfigParser()
    parser.read(config_path)

    return {s: dict(parser.items(s)) for s in parser.sections()}