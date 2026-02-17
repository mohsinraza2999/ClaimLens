import pandas as pd
from src.inference.core.model_loader import load_model
from src.utils.schemas import APIResponse

class Prediction:
    def __init__(self,data:APIResponse):
        self.data=self._formate_data(data)
        self.model=load_model()

    def _formate_data(self,data:APIResponse):
        """Convert API input to DataFrame suitable for the pipeline."""
        # Convert Pydantic / dict to DataFrame
        try:
            return pd.DataFrame(data.model_dump(),index=[0])
        except Exception as e:
            raise ValueError("bad data, data cannot be formated!")
        
    def predict_class(self)-> str:

        prediction = self.model.predict(self.data)

        return "claim" if prediction==1 else "opinion"
        


    
