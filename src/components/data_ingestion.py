import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfiguration:
    train_data_path:str=os.path.join('artifacts', 'train.csv')
    test_data_path:str=os.path.join('artifacts', 'test.csv')
    raw_data_path:str=os.path.join('artifacts', 'raw_data.csv')
    actual_dataset_path:str=os.path.join('notebooks/data', 'heart_disease.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfiguration()
    
    def initiate_data_ingestion(self):
        logging.info('Starting Data ingestion')

        try:            
            df = pd.read_csv(self.ingestion_config.actual_dataset_path)
            
            logging.info(f'Read data from {self.ingestion_config.actual_dataset_path}')
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info(f'Create raw data at {self.ingestion_config.raw_data_path}')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info(f'Create train data at {self.ingestion_config.train_data_path}')

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f'Create test data at {self.ingestion_config.test_data_path}')

            logging.info(f'Data Ingestion completed.')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as ex:
            logging.error('Exception occured while Data ingestion')
            raise CustomException(ex, sys)

