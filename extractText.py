import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

if __name__ == "__main__":
    pdf_path = "path_to_your_pdf.pdf"  # Replace with your PDF file path
    text = extract_text_from_pdf(pdf_path)
    print(text)
