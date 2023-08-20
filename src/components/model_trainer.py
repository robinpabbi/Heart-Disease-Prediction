import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.utils import save_object, evaluate_model
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfiguration:
    trained_model_file_path:str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self) -> None:
        self.model_trainer_config: ModelTrainerConfiguration = ModelTrainerConfiguration()

    def train_model(self, train_array, test_array):
        try:
            logging.info('Splitting Independent and Dependent Variables from train and test data.')
            
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            # Dictionary of all the models needed to be tried
            models = {
                'LogisticRegression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'SVC': SVC(),
                'NB': GaussianNB()
            }

            eval_report = evaluate_model(X_train, X_test, y_train, y_test, models)

            logging.info(f'Model report: {eval_report}')

            # Get the key of the dictionary with max value
            best_model_name = max(eval_report, key= lambda x: eval_report[x])

            best_model_score = eval_report[best_model_name]

            logging.info(f'Best Model Found, Model Name: {best_model_name} roc_auc_score: {best_model_score}')

            save_object(self.model_trainer_config.trained_model_file_path,
                        models[best_model_name]
                        )

        except Exception as ex:
            logging.error('Error occured while training model.')
            raise CustomException(ex, sys)