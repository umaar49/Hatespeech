import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Page config
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫")


# Model loading
@st.cache_resource
def load_model():
    MODEL_DIR = "Umaar49/hatespeech-detector"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model


tokenizer, model = load_model()


def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.max(probabilities).item()
    return predicted_class, confidence, probabilities.tolist()[0]


# UI
st.title("🚫 Hate Speech Detection")
st.write("Enter text to classify as **Normal**, **Offensive**, or **Hate Speech**")

# Sidebar with examples
with st.sidebar:
    st.header("💡 Try Examples")
    examples = {
        "Normal": "Have a great day!",
        "Offensive": "You're so dumb",
        "Hate Speech": "I hate people of your religion"
    }
    for label, example in examples.items():
        if st.button(f"{label}: '{example}'"):
            st.session_state.user_input = example

# Main input
user_input = st.text_area(
    "✍️ Enter your text here:",
    value=st.session_state.get('user_input', ''),
    height=1
)

if st.button("🔍 Analyze Text", type="primary"):
    if user_input.strip():
        pred, confidence, all_probs = predict_text(user_input)

    
        labels = {
            0: "🚫 Hate Speech",
            1: "😠 Offensive",
            2: "😊 Normal"
        }

        result = labels.get(pred, "Unknown")

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction:")
            st.success(f"**{result}**")
        with col2:
            st.subheader("Confidence:")
            st.info(f"**{confidence:.2%}**")

     
        st.subheader("📊 Detailed Breakdown:")
        for class_idx in range(3): 
            label = labels[class_idx]
            prob = all_probs[class_idx]
            st.progress(prob, text=f"{label}: {prob:.2%}")

    else:
        st.warning("Please enter some text to analyze.")
