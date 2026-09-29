import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.vehicle_data import Proj1Data


class DataIngestion:

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        Initializes the DataIngestion component.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)


    def export_data_into_feature_store(self) -> DataFrame:
        """
        Fetches data from SQLite and stores it as a CSV
        in the feature store.

        Returns:
            DataFrame: The complete dataset.
        """
        try:
            logging.info("Exporting data from SQLite")

            my_data = Proj1Data()

            dataframe = my_data.export_table_as_dataframe(
                table_name=self.data_ingestion_config.table_name
            )

            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            dir_path = os.path.dirname(feature_store_file_path)

            os.makedirs(
                dir_path,
                exist_ok=True
            )

            logging.info(
                f"Saving exported data into feature store: "
                f"{feature_store_file_path}"
            )

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True
            )

            return dataframe

        except Exception as e:
            raise MyException(e, sys)


    def split_data_as_train_test(
        self,
        dataframe: DataFrame
    ) -> None:
        """
        Splits the dataset into training and testing sets
        and saves them as CSV files.
        """

        logging.info(
            "Entered split_data_as_train_test method"
        )

        try:

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info(
                "Performed train-test split"
            )

            dir_path = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )

            os.makedirs(
                dir_path,
                exist_ok=True
            )

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info(
                "Train and test files exported successfully"
            )

        except Exception as e:
            raise MyException(e, sys) from e


    def initiate_data_ingestion(
        self
    ) -> DataIngestionArtifact:
        """
        Executes the complete data ingestion process.

        Returns:
            DataIngestionArtifact:
            Paths of training and testing datasets.
        """

        logging.info(
            "Entered initiate_data_ingestion method"
        )

        try:

            dataframe = self.export_data_into_feature_store()

            logging.info(
                "Data successfully fetched from SQLite"
            )

            self.split_data_as_train_test(dataframe)

            logging.info(
                "Train-test split completed"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=
                self.data_ingestion_config.training_file_path,

                testing_file_path=
                self.data_ingestion_config.testing_file_path
            )

            logging.info(
                f"Data ingestion artifact: "
                f"{data_ingestion_artifact}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e