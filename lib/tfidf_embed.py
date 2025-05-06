import numpy as np
import os
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from embedding_utils import chunk_text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data

BATCH_SIZE = 100
EMBED_DIM = 768  # Adjust as needed

def check_nltk_data():
    """Check if required NLTK data is downloaded, if not download it."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

# Check and download NLTK data if needed
check_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Enhanced preprocessing for TF-IDF in RAG:
    - Lowercase
    - Remove non-alphabetic characters
    - Tokenize
    - Remove stopwords
    - Lemmatize
    - Rejoin to string
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

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

def stream_jsonl_text(jsonl_file):
    """Stream documents from a JSONL file."""
    with open(jsonl_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            yield preprocess_text(doc['text'])

def stream_jsonl_documents(jsonl_file):
    """Stream documents from a JSONL file."""
    with open(jsonl_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            yield doc['doc_id'], preprocess_text(doc['text'])

def load_vectorizer_from_config(model_config_file, jsonl_file):
    """
    Load a pre-trained TF-IDF vectorizer from a model configuration file.
    
    Args:
        model_config_file: Path to the model configuration file
        jsonl_file: Path to the JSONL file containing documents (used if creating a new vectorizer)
        
    Returns:
        tfidf_vectorizer: The loaded or newly created TF-IDF vectorizer
    """
    if not model_config_file or not os.path.exists(model_config_file):
        print("No model config file provided or file does not exist. Creating new vectorizer.")
        return None
        
    print(f"Loading pre-trained vectorizer from config file: {model_config_file}")
    with open(model_config_file, 'r') as f:
        config = json.load(f)
    
    if 'model_prefix' not in config:
        print("Warning: 'model_prefix' not found in config file. Creating new vectorizer.")
        return None
        
    model_prefix = config['model_prefix']
    vocab_path = f"{model_prefix}_vocab.npy"
    idf_path = f"{model_prefix}_idf.npy"
    
    if not os.path.exists(vocab_path) or not os.path.exists(idf_path):
        print(f"Warning: Vectorizer components not found at {model_prefix}. Creating new vectorizer.")
        return None

    print(f"Loading vocabulary from {vocab_path}")
    vocabulary = np.load(vocab_path, allow_pickle=True)
    
    print(f"Loading IDF values from {idf_path}")
    idf_values = np.load(idf_path)
    
    # Create vocabulary dictionary
    vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}
    
    # Initialize TF-IDF vectorizer with loaded vocabulary and IDF values
    print("Initializing TF-IDF vectorizer with loaded components...")
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_dict)
    tfidf_vectorizer.idf_ = idf_values
    
    print("Pre-trained vectorizer loaded successfully")
    return tfidf_vectorizer

def create_new_vectorizer(vectorizer_file, jsonl_file):
    """
    Create a new TF-IDF vectorizer from the input documents.
    
    Args:
        vectorizer_file: Base path for saving vectorizer components
        jsonl_file: Path to the JSONL file containing documents
        
    Returns:
        tfidf_vectorizer: The newly created TF-IDF vectorizer
    """
    # Step 1: Build vocabulary  
    print("Building vocabulary...")
    vocab_vectorizer = CountVectorizer(max_features=EMBED_DIM)
    vocab_vectorizer.fit(stream_jsonl_text(jsonl_file))

    # Step 2: Initialize TF-IDF vectorizer with fixed vocabulary
    print("Creating TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_vectorizer.vocabulary_)
    tfidf_vectorizer.fit(stream_jsonl_text(jsonl_file))

    # Save vectorizer components
    print("Saving vectorizer components...")
    vocabulary = np.array(list(tfidf_vectorizer.vocabulary_.keys()))
    np.save(f"{vectorizer_file}_vocab.npy", vocabulary)
    idf_values = tfidf_vectorizer.idf_
    np.save(f"{vectorizer_file}_idf.npy", idf_values)
    
    return tfidf_vectorizer

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

def process_jsonl_documents_tfidf(jsonl_file, output_file, chunk_size=-1, chunk_overlap=-1, model_config_file=None, chunk_method="fixed"):

    """
    Process a JSONL file containing documents and generate TF-IDF embeddings.
    
    Args:
        jsonl_file: Path to the JSONL file containing documents
        output_file: Path to the output file for embeddings
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        model_config_file: Path to a JSON file containing model configuration
    """
    vectorizer_file = output_file.replace(".jsonl", "")
    
    # Try to load a pre-trained vectorizer from the config file
    tfidf_vectorizer = load_vectorizer_from_config(model_config_file, jsonl_file)
    
    # If no pre-trained vectorizer was loaded, create a new one
    if tfidf_vectorizer is None:
        tfidf_vectorizer = create_new_vectorizer(vectorizer_file, jsonl_file)

    # Step 3: Process batches, collect results
    print("Processing batches and accumulating TF-IDF matrix...")
    all_tfidf = []
    all_metadata = []

    batch = []
    batch_metadata = []

    chunk_iterator = chunk_text(chunk_method)
    for doc_id, text in stream_jsonl_documents(jsonl_file):
        chunk_index = 0
        for chunk in chunk_iterator(text, chunk_size, chunk_overlap):
            batch.append(chunk)
            batch_metadata.append({"doc_id": doc_id, "text": chunk, "chunk_index": chunk_index})
            chunk_index += 1
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
                "chunk_index": metadata['chunk_index'],
                "chunk_text": metadata['text']
            }
            out_f.write(json.dumps(output_data) + '\n')

    print(f"\nSaved final TF-IDF matrix with shape {final_matrix.shape} to {output_file}")
    print(f"Saved vectorizer components to {vectorizer_file}_vocab.npy and {vectorizer_file}_idf.npy")
