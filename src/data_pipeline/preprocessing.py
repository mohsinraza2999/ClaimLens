import pandas as pd
from pathlib import Path
from src.data_pipeline.data_loader import ReadData
from src.utils.configs_loader import Configurations
from src.utils.logger import logger
class PreprocessData:
    def __init__(self, df:pd.DataFrame)->None:
        self.df: pd.DataFrame =df

    def invalid_handler(self)->None:
        for col in self.df.columns:
            (logger.info if logger else print)(f"handling invalid values in {col} column")
            self.df[col]=self._remove_invalid(self.df[col])

    def missing_handler(self)->None:
        null=False
        null_meta=self.df.isnull().sum()
        (logger.info if logger else print)(f"missing values details are: \n {null_meta}")
        for cols, nullv in null_meta.items():
            # --- Handle missing values ---
            if nullv>0:
                null=True
                break
                # Uncomment the below line to Replace numerical missing values with median
                # self.df[cols]=self._int_value_handler(self.df[cols])

        if null:
            self.df = self.df.dropna(axis=0)
            (logger.info if logger else print)("Drop null values")
        
    def duplicate_handler(self)->None:
        # --- Handle duplicate values ---
        if self.df.duplicated().any():
            (logger.info if logger else print)("Drop Duplicates")
            self.df = self.df.drop_duplicates()
            
    def outlier_handler(self,cols)->None:
        # --- Handle missing values ---
        (logger.info if logger else print)("checking for outliers")
        for col in cols:
            percentile25 = self.df[col].quantile(0.25)
            percentile75 = self.df[col].quantile(0.75)

            iqr = percentile75 - percentile25
            upper_limit = percentile75 + 2.5 * iqr

            self.df.loc[self.df[col] > upper_limit, col] = upper_limit

    def drop_unwanted(self, cols: list)->None:
        # --- Handle unnessary columns ---
        self.df = self.df.drop(columns=cols, axis=1)
        (logger.info if logger else print)("Drop unwanted columns")

    def save(self, name, path)->None:
        # --- Saves the processed data ---
        if not Path(path).exists():
            Path(path).mkdir(parents=True, exist_ok=True)
        self.df.to_csv(Path(path,name))
        (logger.info if logger else print)("👍 saved processed data")

    def _int_value_handler(value:pd.Series)-> pd.Series:
        if value.dtype in [int,float]:
            return value.fillna(value.median) # Replace missing age with median or change accordingly
        return value
    
    def _remove_invalid(series:pd.Series)-> pd.Series:
        # --- Detect invalid values of numericals and fill with nan ---
        if series.dtype in [int,float]:
            return series.mask(series < 0)  # Mark negative ages as NaN
        return series



def main():
    data_cfg=Configurations().load_data_cfg()
    read_obj=ReadData(data_cfg['raw']['path'])

    prep_obj = PreprocessData(read_obj.read(data_cfg['raw']['name']))

    (logger.info if logger else print)("data preprocessing starts 🎠")

    prep_obj.invalid_handler()

    prep_obj.missing_handler()

    prep_obj.duplicate_handler()

    prep_obj.outlier_handler(data_cfg['outllier_col'])

    prep_obj.drop_unwanted(data_cfg['drop'])

    prep_obj.save(data_cfg['processed'])

    (logger.info if logger else print)("data preprocessing finished 🚩")


if __name__=="__main__":
    main()

