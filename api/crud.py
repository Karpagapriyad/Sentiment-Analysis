from db import create_connection
from psycopg2 import sql
import pandas as pd

def create_predictions_table(conn):
    cursor = conn.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS prediction(
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
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


def get_sentences(conn,tweet):
    cursor = conn.cursor()
    get_data = f"""
    SELECT tweets FROM prediction;
"""
    cursor.execute(get_data,(tweet))
    tweets = [row[0] for row in cursor.fetchall()]
    return tweets


def get_prediction(conn, name):
    cursor = conn.cursor()
    get_data_name = """
    SELECT id, username, tweets, prediction, model_choice FROM prediction WHERE username = %s;
    """
    cursor.execute(get_data_name, (name,))
    rows = cursor.fetchall()
    cursor.close()
    return rows