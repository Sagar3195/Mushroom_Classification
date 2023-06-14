import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from dataclasses import dataclass 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        

        try:
            categorical_columns= ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 
             'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type', 
             'spore-print-color', 'population', 'habitat']


            cat_pipeline = Pipeline(
                steps= [
                    ('one_hot_encoder', OneHotEncoder(handle_unknown= 'ignore', sparse_output= False)),
                    ('scaler', StandardScaler(with_mean= False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor= ColumnTransformer(
                [('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            preprocesssing_obj = self.get_data_transformer_object()

            target_column_name = 'class'

            #Dropping static column from train and test data
            static_column = 'veil-type'
            #Here we replace missing value '?' with most frequent values in stalk root features
            train_df['stalk-root'] = train_df['stalk-root'].replace('?', 'b')
            test_df['stalk-root'] = test_df['stalk-root'].replace('?', 'b')

            #Encoding Target Variable
            train_df[target_column_name] = train_df[target_column_name].map({"e":0, "p":1})
            test_df[target_column_name] = test_df[target_column_name].map({"e":0, "p":1})
            
            input_feature_train_df= train_df.drop(columns= [target_column_name, static_column], axis= 1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns= [target_column_name, static_column], axis= 1)
            target_feature_test_df= test_df[target_column_name]
            

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr= preprocesssing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr= preprocesssing_obj.transform(input_feature_test_df) 

            
            # input_feature_train_arr= scaler.fit_transform(input_feature_train_arr)
            # input_feature_test_arr= scaler.transform(input_feature_test_arr) 

            # train_arr= pd.DataFrame(data= input_feature_train_arr, columns= one_hot_encoder.get_feature_names_out())
            # test_arr= pd.DataFrame(data= input_feature_test_arr, columns= one_hot_encoder.get_feature_names_out())  


            # train_arr[target_column]= target_feature_train_df
            # test_arr[target_column]= target_feature_test_df

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 

            # print(train_arr)
            print()
            # print(test_arr)

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocesssing_obj
            )   

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )

        except Exception as e:
            raise CustomException(e, sys)

