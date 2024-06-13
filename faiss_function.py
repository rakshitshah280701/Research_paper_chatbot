from summarizeText import summarize_text, embed_text
from sklearn.cluster import KMeans
import numpy as np
import time

def cluster_and_summarize(text, huggingface_token):
    model_name = "mrm8488/t5-small-finetuned-gc-3-last-g"
    max_chunk_length = 512

    # Split the text into chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    start_time = time.time()
    embeddings = embed_text(chunks, model_name, huggingface_token)
    print(f"Text embedded in {time.time() - start_time:.2f} seconds")

    num_clusters = min(5, len(embeddings))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    summary_chunks = []
    for center in cluster_centers:
        closest_index = np.argmin(np.linalg.norm(embeddings - center, axis=1))
        summary_chunks.append(chunks[closest_index])

    start_time = time.time()
    summary = summarize_text(" ".join(summary_chunks), model_name, huggingface_token)
    print(f"Text summarized in {time.time() - start_time:.2f} seconds")

    return summary
