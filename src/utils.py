import os
import sys
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True) 
        
        with open(file_path, 'wb') as file_obj:
            logging.info("Dumping pickle file.")
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


