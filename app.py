import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved models and tokenizer
model_lstm = load_model("src/models/lstm_model.keras")
model_cnn = load_model("src/models/cnn_model.keras")
tokenizer = pickle.load(open("src/models/tokenizer.pkl", "rb"))

st.title("🧠 Fake Review Detection App")
st.write("Compare LSTM vs CNN deep learning models on your review text.")

review = st.text_area("Enter a product review:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=100, padding='post')

    pred_lstm = model_lstm.predict(pad)[0][0]
    pred_cnn = model_cnn.predict(pad)[0][0]

    st.subheader("Results:")
    st.write(f"**LSTM Prediction:** {'Fake' if pred_lstm < 0.5 else 'Genuine'} ({pred_lstm:.2f})")
    st.write(f"**CNN Prediction:** {'Fake' if pred_cnn < 0.5 else 'Genuine'} ({pred_cnn:.2f})")
