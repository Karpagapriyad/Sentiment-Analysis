# Sentiment-Analysis
### Introducion:
This project delves into sentiment analysis, a technique for extracting sentiment (joy, anger, sadness, fear, happiness, suprised) from text data. We've implemented two machine learning models – Long Short-Term Memory (LSTM) and Support Vector Machine (SVM) – trained from scratch to analyze the sentiment of textual content.

### Key Functionalities:
1. **Model Training**: We've meticulously trained both LSTM and SVM models from scratch, meticulously selecting and preparing training data to ensure optimal performance.
2. **API Endpoints**: Three robust API endpoints have been created to facilitate various sentiment analysis tasks:
3. **Prediction**: This endpoint allows users to submit text data and receive its predicted sentiment classification.
4. **Past Predictions**: Users can retrieve past sentiment analysis results stored in the database for reference or analysis.
5. **Similarity Tweets**: This innovative endpoint identifies tweets from the database that are semantically similar to a provided query tweet, potentially revealing related sentiment trends.
6. **Streamlit User Interface**: A user-friendly Streamlit UI has been developed to provide a convenient interface for interacting with the API endpoints. Users can effortlessly submit text data, view past predictions, and explore similar tweets.
7. **Database Storage**: We've integrated a database to persistently store sentiment analysis results, enabling efficient retrieval of past predictions and facilitating further analysis.

### Installation Guide:

1. **Clone the Repository:** `git clone (https://github.com/Karpagapriyad/Sentiment-Analysis)`
2. **Navigate to Project Directory:** `cd Streamlit-Analysis`
3. **Install Dependencies:** `pip install -r requirements.txt`
4. **Start FastAPI Server:** `cd api then uvicorn main:app --reload`
5. **Launch Streamlit App:** `cd streamlit then streamlit run main.py`
