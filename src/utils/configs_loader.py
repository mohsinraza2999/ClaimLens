import yaml
from pathlib import Path
from typing import Dict
import os
from src.utils.path_validate import path_exist

class Configurations:
    def __init__(self, CONFIG_PATH:Path=Path(os.getenv("CONFIG_DIR"),"configs"))->None:
        
        self._DATA_CFG=path_exist(CONFIG_PATH / "data_config.yaml")
        self._TRAIN_CFG=path_exist(CONFIG_PATH / "train_config.yaml")

    def load_data_cfg(self)-> Dict[str,str]:
        with open(self._DATA_CFG,'r') as f:
            return yaml.safe_load(f)
        
    def load_train_cfg(self)-> Dict[str,str]:
        with open(self._TRAIN_CFG,'r') as f:
            return yaml.safe_load(f)
    



        
        