import streamlit as st
from PyPDF2 import PdfReader
import summarizeText

st.title("Research Paper Summarizer")

def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        cleaned_text = summarizeText.clean_text(text)
        abstract, conclusion = summarizeText.extract_abstract_and_conclusion(cleaned_text)
        st.subheader("Extracted Abstract:")
        st.text_area("Abstract:", value=abstract, height=150)
        st.subheader("Extracted Conclusion:")
        st.text_area("Conclusion:", value=conclusion, height=150)

        if st.button("Add to Index"):
            combined_text = f"Abstract: {abstract}\n\nConclusion: {conclusion}"
            summary = summarizeText.chunk_and_summarize(combined_text)
            st.success(f"Added to index with summary: {summary}")
