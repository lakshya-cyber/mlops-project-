import os
import sys
import pandas as pd

from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

from src.exception import MyException
from src.constants import TARGET_COLUMN, MODEL_FILE_NAME
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e


    def get_best_model(self) -> Optional[object]:
        """
        Loads the current production/local registry model
        if one already exists.
        """
        try:
            current_model_path = os.path.join(
                "model_registry",
                MODEL_FILE_NAME
            )

            if os.path.exists(current_model_path):
                logging.info(
                    f"Existing model found at: {current_model_path}"
                )

                return load_object(
                    file_path=current_model_path
                )

            logging.info("No existing production model found.")
            return None

        except Exception as e:
            raise MyException(e, sys) from e


    def _map_gender_column(self, df):
        logging.info("Mapping Gender column")

        df["Gender"] = df["Gender"].map({
            "Female": 0,
            "Male": 1
        }).astype(int)

        return df


    def _create_dummy_columns(self, df):
        logging.info("Creating dummy columns")

        return pd.get_dummies(
            df,
            drop_first=True
        )


    def _rename_columns(self, df):

        df = df.rename(columns={
            "Vehicle_Age_< 1 Year":
                "Vehicle_Age_lt_1_Year",

            "Vehicle_Age_> 2 Years":
                "Vehicle_Age_gt_2_Years"
        })

        for col in [
            "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_gt_2_Years",
            "Vehicle_Damage_Yes"
        ]:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df


    def _drop_id_column(self, df):
        """
        Drop dataset identifier.
        """
        logging.info("Dropping id column")

        if "id" in df.columns:
            df = df.drop(
                columns=["id"]
            )

        return df


    def evaluate_model(self) -> EvaluateModelResponse:

        try:

            test_df = pd.read_csv(
                self.data_ingestion_artifact.testing_file_path
            )

            x = test_df.drop(
                TARGET_COLUMN,
                axis=1
            )

            y = test_df[TARGET_COLUMN]

            logging.info(
                "Test data loaded for model evaluation"
            )

            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = load_object(
                file_path=
                self.model_trainer_artifact.trained_model_file_path
            )

            logging.info("New trained model loaded.")

            trained_model_f1_score = (
                self.model_trainer_artifact
                .metric_artifact
                .f1_score
            )

            logging.info(
                f"New model F1 score: "
                f"{trained_model_f1_score}"
            )

            best_model_f1_score = None

            best_model = self.get_best_model()

            if best_model is not None:

                logging.info(
                    "Evaluating existing production model"
                )

                y_hat_best_model = best_model.predict(x)

                best_model_f1_score = f1_score(
                    y,
                    y_hat_best_model
                )

                logging.info(
                    f"Existing model F1: "
                    f"{best_model_f1_score}"
                )

            current_best_score = (
                0
                if best_model_f1_score is None
                else best_model_f1_score
            )

            difference = (
                trained_model_f1_score
                - current_best_score
            )

            is_model_accepted = (
                best_model is None
                or difference
                >= self.model_eval_config.changed_threshold_score
            )

            result = EvaluateModelResponse(
                trained_model_f1_score=
                trained_model_f1_score,

                best_model_f1_score=
                current_best_score,

                is_model_accepted=
                is_model_accepted,

                difference=
                difference
            )

            logging.info(
                f"Model evaluation result: {result}"
            )

            return result

        except Exception as e:
            raise MyException(e, sys) from e


    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifact:

        try:

            logging.info(
                "Initialized Model Evaluation Component"
            )

            response = self.evaluate_model()

            current_model_path = os.path.join(
                "model_registry",
                MODEL_FILE_NAME
            )

            model_evaluation_artifact = (
                ModelEvaluationArtifact(

                    is_model_accepted=
                    response.is_model_accepted,

                    changed_accuracy=
                    response.difference,

                    current_model_path=
                    current_model_path,

                    trained_model_path=
                    self.model_trainer_artifact
                    .trained_model_file_path
                )
            )

            logging.info(
                f"Model evaluation artifact: "
                f"{model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e