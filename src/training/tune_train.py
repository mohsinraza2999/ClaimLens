from pathlib import Path
from joblib import dump, load
import json
import pandas as pd
from src.training.training_pipeline import TrainPipeline
from src.training.model_tuning import Tuning
from src.utils.configs_loader import Configurations
from src.data_pipeline.splitter import split_data
from src.utils.eval_metrics import eval_metrics_scores
from src.utils.make_results import organize_result, save_results, save_best
class TuneTrain(Tuning):
    def __init__(self):
        self._TRAIN_CFG=Configurations().load_train_cfg()
        (self.X_trn, self.y_trn), (self.X_test, self.y_test)=split_data(self._TRAIN_CFG['data'])
        self.best_models=[]
        super.__init__(self.X_trn, self.y_trn,self._TRAIN_CFG['tuning'])

    def tune(self, pipeline_obj):

        for model_name in self._TRAIN_CFG['models']:
            pipeline= pipeline_obj.pipeline(model_name)

            gride_object=self.model_tuning(model_name, pipeline)

            results=organize_result(model_name,gride_object, self._TRAIN_CFG['tuning']['refit'])
            self.eval_best_estimator(model_name,gride_object.best_estimator_, results)

            save_best(gride_object.best_estimator_, model_name,
                      gride_object.best_params_, gride_object.best_score_)
            
            self.best_models.append({"name":model_name, "score":gride_object.best_score_,
                                     "best_params":gride_object.best_params_})

    def eval_best_estimator(self, name, hpt_model, results):

        predicts=hpt_model.predict(self.X_test)

        test_results=eval_metrics_scores(name, predicts, self.y_test)

        save_results(results, test_results)

    def train_best(self, path, results):
        best_model = load(path)
        #(logger.info if logger else print)("Best Model Loaded From the given Path")
        
        best_model.fit(pd.concat([self.X_trn, self.X_test]), pd.concat([self.y_trn, self.y_test]))
        #(logger.info if logger else print)("Refit on all Data")
        model_dir=self._TRAIN_CFG["model_dir"]
        # Save final production model
        dump(best_model, Path(model_dir, "models", "production_model.joblib"))
        Path("", "results", "winner.json").write_text(json.dumps(results, indent=2))
        #(logger.info if logger else print)("Production Model Saved")

def tp_main():
    obj = TuneTrain()
    obj.tune(TrainPipeline())
    
    # Selecting the best by primary metric
    obj.best_models.sort(key=lambda r: r["best_score"], reverse=True)
    production = obj.best_models[0]
    model_dir=obj._TRAIN_CFG["model_dir"]
    best_model_path = Path(model_dir,"models", f"{production['model_name']}_best.joblib")

    obj.train_best(best_model_path, production)
    


if __name__=="__main__":
    tp_main()