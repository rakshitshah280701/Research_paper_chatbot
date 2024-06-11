import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def setup_faiss_index(embedded_texts):
    dimension = embedded_texts.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedded_texts)
    return index

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    texts = ["example text 1", "example text 2"]  # Replace with your texts
    embedded_texts = np.vstack([embed_text(text, model, tokenizer) for text in texts])
    index = setup_faiss_index(embedded_texts)

    print("FAISS index created and texts added.")
