import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            target_column= 'class'

            #Dropping static column from train and test data
            static_column = 'veil-type'
            

            train_df[target_column] = train_df[target_column].replace({"e":0, "p":1})

            input_feature_train_df= train_df.drop(columns= [target_column, static_column], axis= 1)
            target_feature_train_df= train_df[target_column]

            test_df[target_column] = test_df[target_column].replace({"e":0, "p":1})

            input_feature_test_df= test_df.drop(columns= [target_column, static_column], axis= 1)
            target_feature_test_df= test_df[target_column]
            
            logging.info("Seaparated target column and static column.")

            logging.info("Applying One Hot Encoding Technique.")
            one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse= False)

            input_feature_train_arr= one_hot_encoder.fit_transform(input_feature_train_df) 
            input_feature_test_arr= one_hot_encoder.transform(input_feature_test_df) 

            logging.info("Applying StandardScaler method to scale dataset.")
            scaler = StandardScaler(with_mean= False)
            
            input_feature_train_arr= scaler.fit_transform(input_feature_train_arr)
            input_feature_test_arr= scaler.transform(input_feature_test_arr) 

            train_arr= pd.DataFrame(data= input_feature_train_arr, columns= one_hot_encoder.get_feature_names_out())
            test_arr= pd.DataFrame(data= input_feature_test_arr, columns= one_hot_encoder.get_feature_names_out()) 

            train_arr[target_column]= target_feature_train_df
            test_arr[target_column]= target_feature_test_df


            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= one_hot_encoder
            )   

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )

        except Exception as e:
            raise CustomException(e, sys)

