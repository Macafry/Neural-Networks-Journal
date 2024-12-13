from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
# from sklearn.metrics import classification_report

import joblib, os, pandas as pd

def get_all_model_names():
        model_names = [
            'GaussianNB',
            'RandomForestClassifier',
            'SVM',
            'RBF',
            'LogisticRegression',
            'KNeighborsClassifier',
            'GradientBoostingClassifier',
            'AdaBoostClassifier',
            'MLPClassifier'
        ]
        return model_names

class SckitLearnWrapper:
    def __init__(self, model_type, number, data, targets):
        self.id = f'{model_type}_{number}'
        self.model_type = model_type
        self.number = number
        
        # bootstrap
        self.data, self.targets = resample(data, targets)
        
        self.model = GridSearchCV(
            get_model(model_type), get_grid_search_dict(model_type), 
            cv=5, 
            n_jobs=-1
        )
        
    def save(self, folder):
        folder = folder.rstrip('/')
        fpath = f'{folder}/{self.id}.pkl'
        joblib.dump(self.model, fpath)
        
        
    def load(self, folder):
        folder = folder.rstrip('/')
        fpath = f'{folder}/{self.id}.pkl'
        self.model = joblib.load(fpath)
        
    def id_saved(self, folder):
        folder = folder.rstrip('/')
        fpath = f'{folder}/{self.id}.pkl'
        return os.path.exists(fpath)
    
    def train(self, save_folder):
        
        if self.id_saved(save_folder):
            self.load(save_folder)
            return self.model

        self.model.fit(self.data, self.targets)
        
        if not os.path.exists(f'{save_folder}/grid_search_results'):
            os.makedirs(f'{save_folder}/grid_search_results')
        
        grid_search_results = pd.DataFrame(self.model.cv_results_)
        grid_search_results.to_csv(f'{save_folder}/grid_search_results/{self.id}.csv')
        
        self.model = self.model.best_estimator_
        self.save(save_folder)
        
        return self.model
        
        
        
        
        
        
def get_model(model_type):
    if model_type == 'GaussianNB':
        return GaussianNB()
    elif model_type == 'RandomForestClassifier':
        return RandomForestClassifier()
    elif model_type == 'SVM':
        return make_pipeline(StandardScaler(), SVC())
    elif model_type == 'RBF':
        return make_pipeline(StandardScaler(), SVC())
    elif model_type == 'LogisticRegression':
        return LogisticRegression()
    elif model_type == 'KNeighborsClassifier':
        return KNeighborsClassifier()
    elif model_type == 'GradientBoostingClassifier':
        return GradientBoostingClassifier()
    elif model_type == 'AdaBoostClassifier':
        return AdaBoostClassifier()
    elif model_type == 'MLPClassifier':
        return MLPClassifier()
        
    
        
def get_grid_search_dict(model_type):
    param_grids = {
        'GaussianNB': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Hyperparameter for GaussianNB
        },
        'RandomForestClassifier': {
            'n_estimators': [100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10],
            'min_samples_leaf': [4],
            'bootstrap': [True]
        },
        'SVM': {
            'svc__C': [0.1, 1, 10, 100],
            'svc__kernel': ['linear'],
        },
        'RBF': {
            'svc__C': [0.1, 1, 10, 100],
            'svc__kernel': ['rbf'],
            'svc__gamma': ['scale', 'auto']  # Important for RBF kernel
        },
        'LogisticRegression': {
            'penalty': ['l2', None],  # Penalty types
            'C': [0.1, 1, 10, 100],  # Regularization strength
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100],
            'learning_rate': [0.01],
            'max_depth': [5, 10, 15],
            'min_samples_split': [10],
            'min_samples_leaf': [4]
        },
        'AdaBoostClassifier': {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME']
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam'],
            'alpha': [0.001],
            'learning_rate': ['adaptive']
        }
    }
    
    return param_grids[model_type]