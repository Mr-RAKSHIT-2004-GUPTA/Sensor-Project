import sys
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.utils.main_utils import MainUtils
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = "artifacts"
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def evaluate_models(self, X, y, models: Dict):
        """Trains models and evaluates their accuracy"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_models(self, x_train, y_train, x_test, y_test) -> Tuple[str, object, float]:
        """Selects the best model from trained ones"""
        try:
            model_report = self.evaluate_models(X=x_train, y=y_train, models=self.models)
            best_model_name = max(model_report, key=model_report.get)
            best_model_object = self.models[best_model_name]
            best_model_score = model_report[best_model_name]

            return best_model_name, best_model_object, best_model_score
        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object, best_model_name, x_train, y_train) -> object:
        """Tunes the best model using GridSearchCV"""
        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(x_train, y_train)

            best_params = grid_search.best_params_
            print(f"Best parameters for {best_model_name}: {best_params}")

            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """Trains, evaluates, and saves the best model"""
        try:
            logging.info("Splitting training and testing data")

            x = train_array[:, :-1]
            y = train_array[:, -1]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            model_report = self.evaluate_models(X=x_train, y=y_train, models=self.models)

            best_model_name = max(model_report, key=model_report.get)
            best_model = self.models[best_model_name]

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            test_model_score = accuracy_score(y_test, y_pred)

            print(f"Best model: {best_model_name} with score: {test_model_score}")

            if test_model_score < 0.5:
                raise Exception("No suitable model found (accuracy below 0.5)")

            logging.info(f"Best model found with accuracy: {test_model_score}")

            model_path = self.model_trainer_config.trained_model_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            self.utils.save_object(file_path=model_path, obj=best_model)
            logging.info(f"Model saved at {model_path}")

            return model_path

        except Exception as e:
            raise CustomException(e, sys)
