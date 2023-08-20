import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def save_object(file_path:str, obj_to_save):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj_to_save, file_obj)

    except Exception as ex:
        logging.error('Error occurred by saving file.')
        raise CustomException(ex, sys)

def evaluate_model(X_train, X_test, y_train, y_test, models:dict) -> dict:
    try:
        # Report dictionary
        report = {}

        for modelKey, modelValue in models.items():
            model = modelValue
            
            logging.info(f'Evaluating Model {modelKey}')

            # train a model on training date
            model.fit(X_train, y_train)

            # Predict on the test data
            y_pred = model.predict(X_test)

            # Retrieve roc_auc_score for the model
            roc_auc_scre = roc_auc_score(y_test, y_pred)

            logging.info(f'roc_auc_score for {modelKey}: {roc_auc_score}')

            # Retrieve accuracy score
            accuracy_scr = accuracy_score(y_test, y_pred)

            logging.info(f'accuracy_score for {modelKey}: {accuracy_scr}')

            # Retrieve Classification Report
            clasfcation_report = classification_report(y_test, y_pred)

            logging.info(f'classification_report for {modelKey}: {clasfcation_report}')

            report[modelKey] = roc_auc_scre

        return report  

    except Exception as ex:
        logging.error('Error occurred while evaluating models.')
        raise CustomException(ex, sys)