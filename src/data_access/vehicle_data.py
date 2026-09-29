import sys
import pandas as pd
import numpy as np

from src.configuration.sqlite_db_connection import SQLiteClient
from src.constants import TABLE_NAME
from src.exception import MyException


class Proj1Data:
    """
    A class to export SQLite table data as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the SQLite client connection.
        """
        try:
            self.sqlite_client = SQLiteClient()
        except Exception as e:
            raise MyException(e, sys)

    def export_table_as_dataframe(
        self,
        table_name: str = TABLE_NAME
    ) -> pd.DataFrame:
        """
        Exports an SQLite table as a pandas DataFrame.

        Parameters
        ----------
        table_name : str
            Name of the SQLite table to export.

        Returns
        -------
        pd.DataFrame
            DataFrame containing table data.
        """
        try:
            print("Fetching data from SQLite")

            query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(
                query,
                self.sqlite_client.connection
            )

            print(f"Data fetched with len: {len(df)}")

            if "id" in df.columns:
                df = df.drop(columns=["id"])

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise MyException(e, sys)