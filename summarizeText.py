from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
from langchain_community.embeddings import HuggingFaceEmbeddings

def summarize_text(text, model_name, huggingface_token, max_chunk_length=512):
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=huggingface_token)

    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=max_chunk_length, truncation=True)
        outputs = model.generate(inputs.input_ids, max_length=150, min_length=30, do_sample=False)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    print(f"Text summarized in {time.time() - start_time:.2f} seconds")
    return " ".join(summaries)

def embed_text(texts, model_name, huggingface_token):
    start_time = time.time()

    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, huggingfacehub_api_token=huggingface_token)
    embeddings = embeddings_model.embed_documents(texts)

    print(f"Text embedded in {time.time() - start_time:.2f} seconds")
    return embeddings
