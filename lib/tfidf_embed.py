import numpy as np
import os
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from embedding_utils import chunk_text

BATCH_SIZE = 100
EMBED_DIM = 768  # Adjust as needed

def preprocess_text(text):
    """
    Preprocess text for TF-IDF vectorization:
    - Convert to lowercase
    - Remove special characters and numbers
    - Normalize whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def stream_document_batches(doc_file, batch_size=100):
    batch = []
    with open(doc_file, "r", encoding="utf-8") as f:
        for line in f:
            file_path = line.strip()
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as doc_file:
                    text = doc_file.read()
                    batch.append(preprocess_text(text))
            else:
                batch.append(preprocess_text(file_path))

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch

def stream_all_documents(doc_file):
    with open(doc_file, "r", encoding="utf-8") as f:
        for line in f:
            file_path = line.strip()
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as doc_file:
                    text = doc_file.read()
                    yield preprocess_text(text)
            else:
                yield preprocess_text(file_path)

def stream_jsonl_documents(jsonl_file):
    """Stream documents from a JSONL file."""
    with open(jsonl_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            yield preprocess_text(doc['text'])

def process_documents(documents_file, output_file, vectorizer_file, embed_dim):
    # Step 1: Build vocabulary
    print("Building vocabulary...")
    vocab_vectorizer = CountVectorizer(max_features=EMBED_DIM)
    vocab_vectorizer.fit(stream_all_documents(documents_file))

    # Step 2: Initialize TF-IDF vectorizer with fixed vocabulary
    print("Creating TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_vectorizer.vocabulary_)

    # Save vectorizer components
    print("Saving vectorizer components...")
    # Save vocabulary as a dictionary mapping terms to indices
    vocabulary = np.array(list(tfidf_vectorizer.vocabulary_.keys()))
    np.save(f"{vectorizer_file}_vocab.npy", vocabulary)
    
    # Save IDF values
    # We need to fit the vectorizer to get the IDF values
    tfidf_vectorizer.fit(stream_all_documents(documents_file))
    idf_values = tfidf_vectorizer.idf_
    np.save(f"{vectorizer_file}_idf.npy", idf_values)

    # Step 3: Process batches, collect results
    print("Processing batches and accumulating TF-IDF matrix...")
    all_tfidf = []

    for i, batch in enumerate(stream_document_batches(documents_file, BATCH_SIZE)):
        tfidf_matrix = tfidf_vectorizer.transform(batch).toarray()
        all_tfidf.append(tfidf_matrix)
        print(f"Processed batch {i} with shape {tfidf_matrix.shape}")

    # Concatenate and save
    final_matrix = np.vstack(all_tfidf)
    np.save(output_file, final_matrix)

    print(f"\nSaved final TF-IDF matrix with shape {final_matrix.shape} to {output_file}")
    print(f"Saved vectorizer components to {vectorizer_file}_vocab.npy and {vectorizer_file}_idf.npy")

def process_jsonl_documents(jsonl_file, output_file, vectorizer_file, embed_dim=768):
    """
    Process a JSONL file containing documents and generate TF-IDF embeddings.
    
    Args:
        jsonl_file: Path to the JSONL file containing documents
        output_file: Path to the output file for embeddings
        vectorizer_file: Base path for saving vectorizer components
        embed_dim: Dimension of the embedding vectors
    """
    import pdb; pdb.set_trace()
    # Step 1: Build vocabulary
    print("Building vocabulary...")
    vocab_vectorizer = CountVectorizer(max_features=embed_dim)
    vocab_vectorizer.fit(stream_jsonl_documents(jsonl_file))

    # Step 2: Initialize TF-IDF vectorizer with fixed vocabulary
    print("Creating TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_vectorizer.vocabulary_)
    tfidf_vectorizer.fit(stream_jsonl_documents(jsonl_file))

    # Save vectorizer components
    print("Saving vectorizer components...")
    vocabulary = np.array(list(tfidf_vectorizer.vocabulary_.keys()))
    np.save(f"{vectorizer_file}_vocab.npy", vocabulary)
    idf_values = tfidf_vectorizer.idf_
    np.save(f"{vectorizer_file}_idf.npy", idf_values)

    # Step 3: Process batches, collect results
    print("Processing batches and accumulating TF-IDF matrix...")
    all_tfidf = []
    all_metadata = []

    batch = []
    batch_metadata = []
    
    for doc in stream_jsonl_documents(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if preprocess_text(json.loads(line)['text']) == doc:
                    batch_metadata.append(json.loads(line))
                    break
        
        batch.append(doc)
        
        if len(batch) >= BATCH_SIZE:
            tfidf_matrix = tfidf_vectorizer.transform(batch).toarray()
            all_tfidf.append(tfidf_matrix)
            all_metadata.extend(batch_metadata)
            batch = []
            batch_metadata = []
            print(f"Processed batch of {len(tfidf_matrix)} documents")

    # Process any remaining documents
    if batch:
        tfidf_matrix = tfidf_vectorizer.transform(batch).toarray()
        all_tfidf.append(tfidf_matrix)
        all_metadata.extend(batch_metadata)

    # Concatenate all matrices
    final_matrix = np.vstack(all_tfidf)

    # Save results
    print("Saving embeddings...")
    with open(output_file, 'w') as out_f:
        for j, (metadata, embedding) in enumerate(zip(all_metadata, final_matrix)):
            output_data = {
                "id": f"tfidf-{metadata['doc_id']}",
                "object": "embedding",
                "embedding": embedding.tolist(),
                "source": jsonl_file,
                "doc_id": metadata['doc_id'],
                "chunk_index": 0,
                "chunk_text": metadata['text']
            }
            out_f.write(json.dumps(output_data) + '\n')

    print(f"\nSaved final TF-IDF matrix with shape {final_matrix.shape} to {output_file}")
    print(f"Saved vectorizer components to {vectorizer_file}_vocab.npy and {vectorizer_file}_idf.npy")
