import os
import sys
import shutil

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import (
    ModelPusherArtifact,
    ModelEvaluationArtifact
)
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        """
        Initializes the ModelPusher component.
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Copies the accepted trained model into
        the local model registry.
        """

        logging.info(
            "Entered initiate_model_pusher method"
        )

        try:
            logging.info(
                "Pushing accepted model to local model registry"
            )

            source_model_path = (
                self.model_evaluation_artifact.trained_model_path
            )

            destination_model_path = (
                self.model_pusher_config.model_file_path
            )

            # Create model registry directory
            os.makedirs(
                os.path.dirname(destination_model_path),
                exist_ok=True
            )

            # Copy trained model
            shutil.copy2(
                source_model_path,
                destination_model_path
            )

            logging.info(
                f"Model copied successfully to: "
                f"{destination_model_path}"
            )

            model_pusher_artifact = ModelPusherArtifact(
                model_registry_path=destination_model_path
            )

            logging.info(
                f"Model pusher artifact: "
                f"{model_pusher_artifact}"
            )

            logging.info(
                "Exited initiate_model_pusher method"
            )

            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e