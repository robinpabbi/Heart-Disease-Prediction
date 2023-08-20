import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':

    # Data Ingestion phase
    data_ingestor=DataIngestion()
    train_data_path, test_data_path=data_ingestor.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    # Data transformation
    data_transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformer.tranform(train_data_path,
                                                        test_data_path)
    
    # Model Training
    model_trainer = ModelTrainer()
    model_trainer.train_model(train_arr, test_arr)


    
