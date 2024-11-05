#import lib and load model
import numpy as np 
import pandas as pd
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()} 

#load model
model = load_model('simple_rnn_imdb.h5')

## decode review 
def decode_review(encoded_review):
    return  " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# preprocess user input 
def preprocess_text(text):
    words = text.lower().split()
    # from docummentation encoding the decoded word 
    encoded_review =  [word_index.get(word,2)+3 for word in words]
    # encoded review not has index numbers (like ohe )
    # encoded review passed as list of list
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction function 
def predict_sentiment(text):
    preprocessed_input = preprocess_text(text)
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'negative'
    
    return sentiment, prediction[0][0]
    
    
    
    
### streamlit app

st.title("IMDB Movie review sentiment analysis")
st.write("enter a movie review to classify it as positive or negative")

#user input
user_input = st.text_area("Movie Review")

# we want to create a button on clicking of which classification should happen
if st.button("Classify"):
    # button clicked then start preprocessing 
    preprocessed_input = preprocess_text(user_input)
    
    #make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0]> 0.5 else "Negative"
    
    #display result
    st.write(f"sentiment :{sentiment}")
    st.write(f"prediction score: {prediction[0][0]}")
else:
    st.write("please enter review")