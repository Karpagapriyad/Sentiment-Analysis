import psycopg2

def create_connection():
    conn = psycopg2.connect(
        dbname="emotional_detection",
        user="postgres",
        password="123456",
        host="localhost",
        port="5432"
    )
    return conn