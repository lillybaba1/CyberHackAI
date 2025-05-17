import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="CyberHackAI", layout="wide")
st.title("🤖 CyberHackAI: Intrusion Detection System Made By Alasan")


model = load_ids_model()

# Load model
@st.cache_resourcest.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="google/flan-t5-base")

chatbot = load_chatbot()



# Sidebar chat
with st.sidebar:
    st.header("💬 CyberHackAI Chatbot")
    chat_history = st.session_state.get("chat_history", [])
    user_input = st.text_input("Ask me anything...")

    if user_input:
        response = chatbot(user_input, max_length=100)[0]['generated_text']
        chat_history.append(("🧑 You", user_input))
        chat_history.append(("🤖 CyberBot", response))
        st.session_state.chat_history = chat_history

    for speaker, text in chat_history[-6:]:
        st.markdown(f"**{speaker}:** {text}")


# Main form
st.subheader("📥 Manually Enter 41 Features")
inputs = [st.number_input(f"Feature f{i}", value=0.0) for i in range(41)]

if st.button("🧠 Predict"):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)
    result = "🔒 Normal" if prediction[0][0] < 0.5 else "⚠️ Intrusion"
    st.success(f"Prediction: {result}")
