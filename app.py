import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="CyberHackAI", layout="wide")
st.title("🤖 CyberHackAI: Intrusion Detection System")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cyberhack_ai_model.keras")

model = load_model()

# Sidebar chat
with st.sidebar:
    st.header("💬 CyberHackAI Chatbot")
    chat_history = st.session_state.get("chat_history", [])
    user_input = st.text_input("Ask me anything...")
    if user_input:
        reply = "🤖 I'm here to help with cybersecurity. Ask about attacks or features!"
        chat_history.append(("🧑 You", user_input))
        chat_history.append(("🤖 CyberBot", reply))
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
