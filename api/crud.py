from db import create_connection
from psycopg2 import sql
import pandas as pd

def create_predictions_table(conn, table_name="prediction"):
    cursor = conn.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        tweets TEXT NOT NULL,
        prediction INT NOT NULL,
        model_choice TEXT NOT NULL
    );
"""
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()


def insert_prediction_table(conn, prediction):
    cursor = conn.cursor()
    username, tweets, prediction, model_choice = (prediction)
    insert_query = f"""
    INSERT INTO prediction (username, tweets, prediction, model_choice)
    VALUES (%s, %s, %s, %s);
"""
    cursor.execute(insert_query, (username, tweets, prediction, model_choice))
    conn.commit()
    cursor.close()


def get_sentences(conn):
    cursor = conn.cursor()
    get_data = f"""
    SELECT tweeets FROM prediction;
"""
    cursor.execute(get_data)
    df = pd.read_sql(get_data, conn)
    return df

def get_prediction(conn, name):
    cursor = conn.cursor()
    get_data = f"""
    SELECT * FROM prediction WHERE username = 'name';
    """
    cursor.execute(get_data)
    df = pd.read_sql(get_data, conn, params=(name,))
    return df