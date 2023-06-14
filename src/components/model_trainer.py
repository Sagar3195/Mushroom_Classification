import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts", 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split train and test dataset.")

            # target_column= 'class'

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # print("Y_train: {0} and \n Y_test: {1}".format(y_train, y_test))
            models = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier()
            }

            params={
                "Logistic Regression": {
                 #   'penalty':['none','l1','l2','elasticnet']                
                },
                "Support Vector Classifier":{
                    'C': [0.1, 1, 10, 100, 1000], 
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                },
                "Decision Tree": {
                   'criterion':['gini', 'entropy'],
                   'max_depth':[2,4,6,8,10,12]
                },
                "KNN":{
                    'n_neighbors': range(1, 21, 2),
                 #   'weights': ['uniform', 'distance'],
                 #   'metric': ['euclidean', 'manhattan', 'minkowski']
                },
            
                "Random Forest":{
                  'n_estimators': [20, 25, 50, 100],
                  'max_features': ['sqrt', 'log2', None],
                  'max_depth': [2,3, 6, 9],
                  'max_leaf_nodes': [2,3, 4, 6,9],
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("Best Model Name: {}".format(best_model))
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            logging.info("Returning accuracy score of model")
            
            acc_score = accuracy_score(y_test, predicted)
            cm = confusion_matrix(y_test, predicted)
            report = classification_report(y_test, predicted)
            
            return "Accuracy score:{0}, \nConfusion Matrix:\n{1} and \nClassification Report:\n{2}".format(acc_score, cm, report)
            
 
        except Exception as e:
            raise CustomException(e,sys)


