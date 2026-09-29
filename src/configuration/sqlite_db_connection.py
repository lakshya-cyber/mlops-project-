import sys
import sqlite3

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_PATH


class SQLiteClient:
    """
    SQLiteClient is responsible for establishing a connection
    to the SQLite database.
    """

    connection = None

    def __init__(self, database_path: str = DATABASE_PATH) -> None:
        """
        Initializes a connection to the SQLite database.

        Parameters
        ----------
        database_path : str
            Path to the SQLite database file.
        """
        try:
            if SQLiteClient.connection is None:
                SQLiteClient.connection = sqlite3.connect(
                    database_path,
                    check_same_thread=False
                )

            self.connection = SQLiteClient.connection

            logging.info(
                f"SQLite connection successful: {database_path}"
            )

        except Exception as e:
            raise MyException(e, sys)