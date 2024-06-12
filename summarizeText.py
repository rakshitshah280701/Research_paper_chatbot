from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

def summarize_text(text, max_chunk_length=1024):
    start_time = time.time()

    # Initialize tokenizer and model with your Hugging Face token
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", token="hf_YOUR_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v2-3b", 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        offload_folder="/Users/rakshitshah/Desktop/Github_projects/offload",
        token="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"
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Split the text into chunks of max_chunk_length
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []

    for idx, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        # Prepare the input for the model
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        
        # Move input IDs and attention mask to the same device as the model
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Generate the summary
        output_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=150, do_sample=False)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        summaries.append(summary)
        print(f"Processed chunk {idx + 1}/{len(chunks)} in {time.time() - chunk_start_time:.2f} seconds")

    # Join the summaries
    return " ".join(summaries)

if __name__ == "__main__":
    sample_text = "Your extracted text here..."  # Replace with your extracted text
    summary = summarize_text(sample_text)
    print(summary)
