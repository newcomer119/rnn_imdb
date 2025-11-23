# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    # Convert to lowercase and remove punctuation, then split
    text = text.lower()
    # Remove punctuation and keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    
    # Encode words: word_index returns 1-9999, we add 3 to get 4-10002
    # IMDB encoding: 0=padding, 1=start, 2=unknown, 3-9999=words
    # We need to cap at 9999 (max_features - 1) and handle unknown words (index 2)
    encoded_review = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3
            # Cap at 9999 (max_features - 1) since embedding layer uses 0-9999
            if idx > 9999:
                encoded_review.append(2)  # Use unknown token if out of range
            else:
                encoded_review.append(idx)
        else:
            encoded_review.append(2)  # Unknown word -> index 2 (don't add 3!)
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding='pre')
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

