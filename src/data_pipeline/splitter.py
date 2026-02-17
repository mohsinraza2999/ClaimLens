from sklearn.model_selection import train_test_split
from src.data_pipeline.data_loader import ReadData

def split_data(info: dict)-> list:
    obj = ReadData(info['path'])
    df = obj.read(info['name'])

    if df.empty:
        ValueError(f"{info['name']} is empty!")
    X=df.drop(columns=info['target'],axis=1)

    y=df[info['target']]

    return train_test_split(X, y, test_size=info['test_size'],
                            random_state=info['random_state'], stratify=y)