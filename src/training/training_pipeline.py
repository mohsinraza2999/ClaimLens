from sklearn.pipeline import Pipeline
from src.model.video_classifer_models import Classifier
from src.data_pipeline.data_loader import load_pipeline
from src.utils.configs_loader import Configurations


class TrainPipeline:
    def __init__(self):
        self._data_config=Configurations().load_data_cfg()

    def pipeline(self, model_name:str):
        obj=Classifier(model_name)
        data_pipeline=load_pipeline(self._data_config['categorical'],
                                    self._data_config['numarical'],
                                    self._data_config['vectorizer'])

        if (obj is None) and (data_pipeline is None):
            raise ValueError(f"data_pipeline or model is None")
        
        return Pipeline(steps=[("preprocess", data_pipeline),
                               ("model", obj.load())])