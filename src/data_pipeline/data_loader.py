import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.path_validate import path_exist

class ReadData():
    def __init__(self,root):
        self.root=path_exist(Path(root))

    def read(self,name: str)-> pd.DataFrame:
        path=path_exist(self.root/name)
        try:
            return pd.read_csv(path)
        except Exception as e:
            LookupError(f"{e}")


def load_pipeline(cat_cols: list[str], numeric_cols: list[str], vec_col: list[str]):

    #(logger.info if logger else print)("Building Data Pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols or []),
            ('tfidf', TfidfVectorizer(max_features=50000,
                                      min_df=3,max_df=0.9,
                                      ngram_range=(1,2),
                                      sublinear_tf=True,norm='l2'
                                      ), vec_col or ['video_transcription_text']),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols or [])
        ],
        remainder="drop"
    )
    return preprocessor