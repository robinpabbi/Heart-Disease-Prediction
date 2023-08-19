import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    data_ingestor=DataIngestion()
    train_data_path, test_data_path=data_ingestor.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    
