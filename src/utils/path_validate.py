from pathlib import Path


def path_exist(path: Path)-> Path:
    if not path.exists():
        FileNotFoundError(f"{path} not Found!")
    return path