import os
import sys
import pandas as pd

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

@dataclass
class PredictionPipelineConfiguration:
    preprocessor_file_path:str=os.path.join('artifacts', 'preprocessor.pkl')
    model_file_path:str=os.path.join('artifacts', 'model.pkl')

class PredictionPipeline:

    def __init__(self) -> None:
        self.prediction_config:PredictionPipelineConfiguration = PredictionPipelineConfiguration()

    def predict(self, features):
        try:
            preprocessor = load_object(self.prediction_config.preprocessor_file_path)

            model = load_object(self.prediction_config.model_file_path)

            scaled_data = preprocessor.transform(features)

            predicted_value = model.predict(scaled_data)

            return predicted_value
        except Exception as ex:
            logging.error('Error occurred while predicting the outcome.')
            CustomException(ex, sys)