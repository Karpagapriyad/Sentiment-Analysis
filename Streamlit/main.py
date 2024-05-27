import streamlit as st
import requests
import json
import pandas as pd



def send_data_to_api(name, tweets, model_type):
    url = 'http://127.0.0.1:8000/predict/'
    data = {
        'name': name,
        'tweet': tweets,
        'model': model_type
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

def main():
    st.title("Sentiment Analysis App")

    page = st.sidebar.selectbox("Choose a page", ["Home", "Past Predictions", "Similar Tweets"])

    if page == "Home":
        st.header("Prediction")
        name = st.text_input('Name')
        tweet = st.text_area('Tweets')
        model = st.selectbox('Type of Model', ['lstm', 'svm'])

        if st.button("Predict"):
            if name and tweet and model:
                response = send_data_to_api(name, tweet, model)
            if response.status_code == 200:
                st.write(response.text)
            else:
                st.error(f'Failed to send data. Status code: {response.status_code}')
        else:
            st.warning('Please fill out all fields.')

    elif page == "Past Predictions":
        st.header("Past Predictions")
        name = st.text_input("Enter your name")
        if st.button("Get Past Predictions"):
            prediction_response = requests.get(f"http://127.0.0.1:8000/past_prediction/", params={"name": name})
            if prediction_response.status_code == 200:
                past_predictions = prediction_response.json()
                # data_dict = json.loads(past_predictions)
                # df = pd.DataFrame(data_dict["prediction_data"], columns=data_dict["id", "username", "tweets", "prediction", "model_choice"])
                st.write(past_predictions)
                # st.dataframe(df)
            else:
                st.error(f'Failed to retrieve past predictions. Status code: {prediction_response.status_code}')


    elif page == "Similar Tweets":
        st.header("Similar Tweets")
        tweet = st.text_input("Enter a sentence")
        if st.button("Get Similar Tweets"):
            similarity_response = requests.get("http://127.0.0.1:8000/find_similar/", params={"tweet": tweet})
            if similarity_response.status_code == 200:
                similar_sentences = similarity_response.json()
                st.write(similar_sentences)
            else:
                st.error(f'Failed to retrieve similar tweets. Status code: {similarity_response.status_code}')

if __name__ == "__main__":
    main()
