import PyPDF2
import time

def extract_text_from_pdf(file):
    start_time = time.time()
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    print(f"Text extracted in {time.time() - start_time:.2f} seconds")
    return text
