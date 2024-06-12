import streamlit as st
import torch
from extractText import extract_text_from_pdf
from summarizeText import summarize_text
from faiss import embed_text, setup_faiss_index
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import login

def main():
    st.title("Research Paper Summarizer and Q&A")
    
    # Hugging Face token login
    huggingface_token = st.text_input("Enter your Hugging Face API token:", type="password")
    if huggingface_token:
        login(token=huggingface_token)
    
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            # Save the uploaded file
            pdf_path = "uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Summarize the text
            summary = summarize_text(text)
            
            # Display the summary
            st.header("Summary")
            st.write(summary)
            
            # Set up FAISS with the text
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            texts = [text]
            embedded_texts = embed_text(text, model, tokenizer)
            index = setup_faiss_index(embedded_texts)
            
            # User input for Q&A
            question = st.text_input("Ask a question about the paper:")
            if question:
                embedded_question = embed_text(question, model, tokenizer)
                D, I = index.search(embedded_question, k=1)
                answer = texts[I[0][0]]
                st.write("Answer:", answer)

if __name__ == "__main__":
    main()
