import streamlit as st
import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Define your Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

# Initialize FAISS index
dimension = 512  # Assuming T5 produces 512-dimensional embeddings
index = faiss.IndexFlatL2(dimension)

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def embed_text(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(inputs)
    # Assuming we're taking the mean of the encoder's last hidden state as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

def add_to_index(text):
    summary = generate_summary(text)
    embedding = embed_text(summary)
    if embedding.shape[1] != dimension:
        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {embedding.shape[1]}")
    index.add(np.array(embedding))
    return summary

def search_index(query):
    embedding = embed_text(query)
    D, I = index.search(np.array(embedding), k=5)
    return I

st.title("Document Summarizer and Search")

# Add text to index
st.header("Add Text to Index")
text = st.text_area("Enter a paragraph to summarize and add to the index:", height=200)
if st.button("Add to Index"):
    if text:
        try:
            summary = add_to_index(text)
            st.success(f"Added to index with summary: {summary}")
        except Exception as e:
            st.error(f"Error: {e}")

# Search in the index
st.header("Search in Index")
query = st.text_area("Enter a query to search similar summaries in the index:", height=200)
if st.button("Search Index"):
    if query:
        try:
            indices = search_index(query)
            st.success(f"Search results: {indices}")
        except Exception as e:
            st.error(f"Error: {e}")
