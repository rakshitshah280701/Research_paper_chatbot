import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

def load_model_and_tokenizer():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def clean_text(text):
    # Implement your text cleaning function here
    return text.replace("\n", " ").replace("\r", "").strip()

def extract_abstract_and_conclusion(text):
    # Use regex to find the Abstract and Conclusion sections
    abstract_pattern = re.compile(r'(Abstract|ABSTRACT)(.*?)(Introduction|INTRODUCTION)', re.DOTALL)
    conclusion_pattern = re.compile(r'(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)(.*?)(References|REFERENCES|Acknowledgments|ACKNOWLEDGMENTS)', re.DOTALL)

    abstract = re.search(abstract_pattern, text)
    conclusion = re.search(conclusion_pattern, text)

    abstract_text = abstract.group(2).strip() if abstract else "Abstract not found."
    conclusion_text = conclusion.group(2).strip() if conclusion else "Conclusion not found."

    return abstract_text, conclusion_text

def summarize_text(text, tokenizer, model, max_length=150, min_length=40, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=num_beams, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def chunk_and_summarize(text, chunk_size=1000):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarize_text(chunk, tokenizer, model) for chunk in chunks]
    return " ".join(summaries)
