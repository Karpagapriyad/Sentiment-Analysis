import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from keras.models import load_model
from db import create_connection
from crud import (create_predictions_table, insert_prediction_table, get_sentences, get_prediction)
from typing import Literal, List
from sentence_transformers import SentenceTransformer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

lstm_model = load_model('../models/lstm_model.h5')
svm_model = joblib.load('../models/svm_model.joblib')
fasttext_model = joblib.load('../models/fasttext_model.joblib')

conn = create_connection()
create_predictions_table(conn)

label_dict = {
    "sadness": 0,
    "joy": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5
}

binary_label_mapping = {
    0: "sadness",  # Let's assume class 0 is mapped to "sadness"
    1: "joy"       # Let's assume class 1 is mapped to "joy"
}
reverse_label_dict = {v: k for k, v in label_dict.items()}

class TweetInput(BaseModel):
    name: str
    tweet: str
    model: Literal['lstm', 'svm']

class SentenceInput(BaseModel):
    sentence: str


model = SentenceTransformer('all-MiniLM-L6-v2')


# def save_db(prediction):
#     username = prediction.get("name", {})
#     tweets = prediction.get("tweet", {})
#     model = prediction.get("model", {})
#     prediction_label = prediction.get("label", {})
#     dt = (username, tweets, prediction_label, model)
#     insert_prediction_table(conn, dt)

tokenizer = Tokenizer()
tokenizer.word_index = {key: int(value) for key, value in tokenizer.word_index.items()}

def preprocess_tweet(tweet):
    sequence = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequence, maxlen=66)
    return padded_sequence

def preprocess_svm(tweet):
 words = tweet.split()
 return np.mean([fasttext_model.wv[word] for word in words if word in fasttext_model.wv] or [np.zeros(100)], axis=0)

def get_similar_tweets(tweet: str, conn):
    tweets = get_sentences(conn,tweet)
    all_tweets = tweets
    all_tweets.append(tweet)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tweets)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    

    top_indices = np.argsort(similarity_scores)[::-1][:10]

    similar_tweets = [tweets[i] for i in top_indices]
    return similar_tweets


@app.post('/predict/')
def predict(data: TweetInput):
    if data.model == 'lstm':
        lstm_sequence = preprocess_tweet(data.tweet)
        lstm_prediction = lstm_model.predict(lstm_sequence)
        lstm_label = int(np.argmax(lstm_prediction)) 
        print(lstm_label)
        label_name = binary_label_mapping[lstm_label]
        dt= (data.name,data.tweet, lstm_label, data.model)
        insert_prediction_table(conn, dt)
        return {"label": lstm_label, "label_name": label_name}
    elif data.model == 'svm':
        svm_sequence = preprocess_svm(data.tweet)
        svm_prediction = 1 if np.dot(svm_model['w'], svm_sequence) + svm_model['b'] > 0 else 0
        label_name = binary_label_mapping[svm_prediction]
        dt= (data.name,data.tweet, svm_prediction, data.model)
        insert_prediction_table(conn, dt)
        return {"label": svm_prediction, "label_name": label_name}
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

@app.get('/past_prediction/')
def get_user_predictions(name: str):
    past_prediction_data = get_prediction(conn,name)
    return past_prediction_data


@app.get("/find_similar/")

def find_similar_tweets(tweet: str):
    similar_tweets = get_similar_tweets(tweet, conn)
    if not similar_tweets:
        return ["No similar tweets found."]
    else:
        return similar_tweets




