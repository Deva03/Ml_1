import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):

        '''
        This Fuction is responsible for all the Data Transformation
        '''

        try:
            numerical_features=["writing_score","reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical column standerd scaling completed.")
            logging.info(f"List of numerical columns are : {numerical_features}")
            logging.info("Categorical column encoding completed.")
            logging.info(f"List of categorical columns are : {categorical_features}")


            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_features),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def intiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the data completed")
            logging.info("Obtaining Preprocessor Object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_columns_name="math_score"
            numerical_features=["writing_score","reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            input_feature_train_df=train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df=train_df[target_columns_name]

            input_feature_test_df=test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df=test_df[target_columns_name]

            logging.info(
                f"Applying Preprocessing object on training and test dataframes."
            )

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)
            
            train_array = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            test_array = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing Objects...")

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_array,
                test_array,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

        

