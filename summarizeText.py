from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import faiss

# Define your Hugging Face token
HUGGINGFACE_TOKEN = "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"

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = T5ForConditionalGeneration.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

# Initialize FAISS index
dimension = 512  # Adjust according to your model's output size
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
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

def add_to_index(text):
    summary = generate_summary(text)
    embedding = embed_text(summary)
    if embedding.shape[1] != dimension:
        raise ValueError(f"Embedding dimension mismatch: expected {dimension}, got {embedding.shape[1]}")
    index.add(np.array(embedding))
    return summary
