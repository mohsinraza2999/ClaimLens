import pytest
import pandas as pd
from src.data_pipeline.data_loader import ReadData, load_pipeline
from src.utils.configs_loader import Configurations
from src.data_pipeline.splitter import split_data
from src.model.video_classifer_models import Classifier
from src.training.training_pipeline import TrainPipeline

@pytest.fixture
def train_config():
    return Configurations().load_train_cfg()

@pytest.fixture
def data_config():
    return Configurations().load_data_cfg()

def test_read(data_config):
    obj=ReadData(data_config['raw']['path'])
    df =obj.read(data_config['raw']['name'])

    assert df is not None
    assert isinstance(df,pd.DataFrame)

def test_spliter(train_config):

    X_train, y_train, X_test, y_test = split_data(train_config['data'])

    assert X_train.__class__ == X_test.__class__ == y_train.__class__ == y_test.__class__==pd.DataFrame

def test_data_pipeline(data_config):
    data_pipeline=load_pipeline(data_config['categorical'],
                                data_config['numarical'],
                                data_config['vectorizer'])

    assert data_pipeline is not None

def test_classifier(train_config):
    for model_name in train_config['models']:
        obj=Classifier(model_name)
        model=obj.load()
        assert model is not None

def test_train_pipeline(train_config):
    obj=TrainPipeline()
    for model_name in train_config['models']:
        pipeline=obj.pipeline(model_name)

        assert pipeline is not None


        
        


