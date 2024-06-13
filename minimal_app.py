import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Hugging Face token
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class_id == 1 else "Negative"

st.title("Text Summarizer")

text = st.text_area("Enter a paragraph to summarize:", height=200)
if st.button("Summarize"):
    if text:
        summary = summarize(text)
        st.write("Summary:", summary)
