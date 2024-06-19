import streamlit as st
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import faiss
import numpy as np

# Import functions from summarize.py
from summarizeText import generate_summary, embed_text, add_to_index

# Define your Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

# Initialize FAISS index
dimension = 512  # Adjust according to your model's output size
index = faiss.IndexFlatL2(dimension)

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.title("Document Summarizer and Search")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text:", text, height=200)

# Add text to index
st.header("Add Text to Index")
if st.button("Add to Index"):
    if text:
        try:
            summary = add_to_index(text)
            st.success(f"Added to index with summary: {summary}")
        except Exception as e:
            st.error(f"Error: {e}")
