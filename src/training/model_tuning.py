from sklearn.model_selection import GridSearchCV

class Tuning:
    def __init__(self, X_train, y_train, params:dict):
        self.tuning_params=params
        self.X_train=X_train
        self.y_train=y_train

    def model_tuning(self, model_name: str, pipeline):
        
        classifier=GridSearchCV(pipeline, self.tuning_params[model_name]["params"],
                                scoring=self.tuning_params['scoring'], cv=self.tuning_params['cv'],
                                refit=self.tuning_params['refit'], n_jobs=self.tuning_params['n_jobs'])
        
        classifier.fit(self.X_train, self.y_train)

        return classifier

        




