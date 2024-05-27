import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from keras.models import load_model
from db import create_connection
from crud import (create_predictions_table, insert_prediction_table, get_sentences, get_prediction)
from typing import Literal
from sentence_transformers import SentenceTransformer, util
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText

app = FastAPI()

lstm_model = load_model('../models/lstm_model.h5')
svm_model = joblib.load('../models/svm_model.joblib')

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

class TweetInput(BaseModel):
    name: str
    tweet: str
    model: Literal['lstm', 'svm']

class SentenceInput(BaseModel):
    sentence: str

class UserPredictionsResponse(BaseModel):
    id: int
    name: str
    tweet: str
    model: str
    prediction: int

model = SentenceTransformer('all-MiniLM-L6-v2')


def save_db(prediction):
    username = prediction.get("name", {})
    tweets = prediction.get("tweet", {})
    model = prediction.get("model", {})
    dt = [username, tweets, model]
    insert_prediction_table(conn, dt)

tokenizer = Tokenizer()
tokenizer.word_index = {key: int(value) for key, value in tokenizer.word_index.items()}

def preprocess_tweet(tweet):
    sequence = tokenizer.texts_to_sequences([tweet])
    padded_sequence = pad_sequences(sequence, maxlen=66)
    return padded_sequence

def compute_sentence_embedding(sentence):
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust `maxlen` based on your model
    embedding = model.predict(padded_sequences)
    return embedding

@app.post('/predict/')
def predict(input_data: TweetInput):
    if input_data.model == 'lstm':
        lstm_sequence = preprocess_tweet(input_data.tweet)
        lstm_prediction = lstm_model.predict(lstm_sequence)
        lstm_label = np.argmax(lstm_prediction)
        # lstm_label = label_dict[lstm_label]
        return lstm_label
    elif input_data.model == 'svm':
        svm_sequence = np.mean([lstm_model.wv[word] for word in input_data.tweet.split() if word in lstm_model.wv] or [np.zeros(100)], axis=0)
        svm_prediction = 1 if np.dot(svm_model['w'], svm_sequence) + svm_model['b'] > 0 else 0
        # svm_prediction = label_dict[svm_prediction]
        return svm_prediction

    else:
        raise HTTPException(status_code=400, detail="Invalid model selected")

@app.get("/user_predictions/")
def user_predictions(name):
    predictions = get_prediction(conn, name)
    if not predictions is None:
        raise HTTPException(status_code=404, detail="User not found or no predictions available")
    return predictions


@app.post("/find_similar/")

def similar_sentences(input_data: SentenceInput):
    df = get_sentences(conn)
    conn.close()

    new_embedding = model.encode(input_data.sentence)

    sentence_embeddings = model.encode(df['sentence'].tolist())

    similarities = cosine_similarity(new_embedding, sentence_embeddings)

    top_indices = similarities.argsort(descending=True)[:10]
    similar_sentences = df.iloc[top_indices]

    similar_sentences_list = []
    for index, row in similar_sentences.iterrows():
        similar_sentences_list.append({
            "sentence": row['sentence'],
            "similarity_score": similarities[index].item()
        })

    return similar_sentences_list




