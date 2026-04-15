import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('sentiment_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

st.title("Sentiment Analysis App")

text = st.text_input("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if text:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=4)  # Match the maxlen from training
        pred = model.predict(padded)[0][0]
        sentiment = "Positive" if pred > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence: {pred:.2f}")
    else:
        st.write("Please enter some text.")