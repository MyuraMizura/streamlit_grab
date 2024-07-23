import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Memuat model dan tokenizer
model = load_model('sentiment_model.keras')
tokenizer = joblib.load('tokenizer.pkl')

# Fungsi untuk memproses dan memprediksi sentimen
def predict_sentiment(text):
    max_len = 100  # Panjang maksimum sequence yang digunakan saat pelatihan
    # Tokenisasi dan padding teks input
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    # Memprediksi sentimen
    prediction = model.predict(padded_sequence)
    # Mengambil skor probabilitas
    positive_prob = prediction[0][0]
    negative_prob = 1 - positive_prob
    # Mengubah prediksi menjadi label
    sentiment = 'Positif' if positive_prob > 0.5 else 'Negatif'
    return sentiment, positive_prob, negative_prob

# Aplikasi Streamlit
st.title('ANALISIS SENTIMEN APLIKASI GRAB INDONESIA')
st.write("Masukkan teks yang ingin dianalisis:")

# Input teks
user_input = st.text_area("Teks", "")

# Tombol untuk menganalisis sentimen
if st.button("Analisis"):
    if user_input:
        sentiment, positive_prob, negative_prob = predict_sentiment(user_input)
        st.write(f"Hasil Analisis Sentimen: **{sentiment}**")
        st.write(f"Probabilitas Positif: **{positive_prob:.2f}**")
        st.write(f"Probabilitas Negatif: **{negative_prob:.2f}**")
    else:
        st.write("Mohon masukkan teks untuk dilakukan analisis.")
