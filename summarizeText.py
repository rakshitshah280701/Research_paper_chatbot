from transformers import pipeline

def summarize_text(text, max_chunk_length=1024):
    # Use DBRX Instruct model from Hugging Face
    summarizer = pipeline("summarization", model="databricks/dbrx-instruct")
    
    # Split the text into chunks of max_chunk_length
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    # Join the summaries
    return " ".join(summaries)



if __name__ == "__main__":
    sample_text = "Your extracted text here..."  # Replace with your extracted text
    summary = summarize_text(sample_text)
    print(summary)
