import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

@dataclass
class DataTransformationConfiguration:
    preprocessor_file_path:str=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfiguration()
    
    def get_transformer(self)->ColumnTransformer:

        try:
            logging.info('Starting Get Data Tranformer.')

            imputer_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('tranformer', imputer_pipeline, ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            ])

            logging.info('Get Data Tranformer completed.')

            return preprocessor

        except Exception as ex:
            logging.error('Exception occurred while getting Data Transformer')
            raise CustomException(ex, sys)
    

    def tranform(self, train_data_path, test_data_path):
        try:
            logging.info('Starting Data Tranformation.')

            training_df = pd.read_csv(train_data_path)

            logging.info(f'Imported training data from {train_data_path}')

            test_df = pd.read_csv(test_data_path)

            target_column_name = 'target'

            drop_columns = [target_column_name]

            #Splitting Training df to independent variables (features) and dependent variables (target)
            training_feature_df = training_df.drop(columns=drop_columns, axis=1)
            training_target_df = training_df[target_column_name]

            #Splitting test_df into independent variables (features) and dependent variables (target)
            test_feature_df = test_df.drop(columns=drop_columns, axis=1)
            test_target_df = test_df[target_column_name]

            logging.info(f'Imported test data from {test_data_path}')

            preprocessor = self.get_transformer()

            logging.info('Successfully retrieved preprocessor, starting processing data.')

            train_feature_arr = preprocessor.fit_transform(training_feature_df)
            
            test_feature_arr = preprocessor.transform(test_feature_df)

            training_processed_data = np.c_[train_feature_arr, np.array(training_target_df)]

            test_processed_data = np.c_[test_feature_arr,  np.array(test_target_df)]

            logging.info('Successfully tranformed training and test data.')

            save_object(self.data_transformation_config.preprocessor_file_path, preprocessor)

            logging.info(f'Successfully saved preprocessor into {self.data_transformation_config.preprocessor_file_path}')
            
            return (
                training_processed_data,
                test_processed_data,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as ex:
            logging.error('Exception occurred while Data Transformation')
            raise CustomException(ex, sys)