import pandas as pd
from pathlib import Path
from joblib import dump
import json

def organize_result(name: str, gride_cv_object, metric):
    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                'recall': 'mean_test_recall',
                'f1': 'mean_test_f1',
                'accuracy': 'mean_test_accuracy',
                }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(gride_cv_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

     # Create table of results
    table = pd.DataFrame({'model': [name],
                        'precision': [precision],
                        'recall': [recall],
                        'f1': [f1],
                        'accuracy': [accuracy],
                        },
                    )
    
    return table




def save_results(t_result, tst_result, path:str='artifacts'):

    if Path(path,"results/results.parquet").exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, t_result, tst_result], axis=0,ignore_index=True)

    else:
        df = pd.concat([t_result, tst_result], axis=0,ignore_index=True)
        Path(path,"results").mkdir(parents=True, exist_ok=True)

    df.to_parquet(Path(path,"results/results.parquet"), index=False)

def save_best(best_model, model_name:str, best_params, best_score, path:str='artifacts')->None:

    Path(path, "results").mkdir(parents=True, exist_ok=True)
    Path(path, "models").mkdir(parents=True, exist_ok=True)

    Path(path, "results", f"{model_name}_best.json").write_text(json.dumps({
        "best_parameters": best_params, "best_score": best_score}, indent=2))

    # Save refit model
    dump(best_model, Path(path, "models", f"{model_name}_best.joblib"))