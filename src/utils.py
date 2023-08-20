import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

def save_object(file_path:str, obj_to_save):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj_to_save, file_obj)

    except Exception as ex:
        logging.error('Error occurred by saving file.')
        raise CustomException(ex, sys)