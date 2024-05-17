import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from gensim.models import FastText
from keras.models import Sequential

def preprocess_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            text, emotion = line.strip().split(';')
            texts.append(text)
            labels.append(emotion)
    return texts, labels

train_texts, train_labels = preprocess_data('data/train.txt')
test_texts, test_labels = preprocess_data('data/test.txt')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_sequence_length = max(len(seq) for seq in train_sequences)
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

fasttext_model = FastText(sentences=[text.split() for text in train_texts],window=5, min_count=1, workers=4, sg=1)

train_vectors = np.array([np.mean([fasttext_model.wv[word] for word in text.split() if word in fasttext_model.wv] or [np.zeros(100)], axis=0) for text in train_texts])
test_vectors = np.array([np.mean([fasttext_model.wv[word] for word in text.split() if word in fasttext_model.wv] or [np.zeros(100)], axis=0) for text in test_texts])


label_dict = {
    "sadness": 0,
    "joy": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5
}
train_labels_numeric = np.array([label_dict[label] for label in train_labels])
test_labels_numeric = np.array([label_dict[label] for label in test_labels])


model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_sequence_length),
    Bidirectional(LSTM(32)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


model.fit(train_data, train_labels_numeric, validation_split=0.2, epochs=10, batch_size=64)

loss, accuracy = model.evaluate(test_data, test_labels_numeric)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)