import json
import os
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import nltk
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embedding_utils import chunk_text

# Ensure NLTK data is available
nltk.data.path.append("./nltk_data")
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("Warning: NLTK data not found. Text preprocessing will be limited.")
    stop_words = set()
    lemmatizer = None

# Constants
BATCH_SIZE = 100
EMBED_DIM = 768
DEFAULT_DATASET_NAME = "tfidf_embeddings"


class TfidfEmbeddingProcessor:
    """Processes documents to create TF-IDF embeddings and saves them as HuggingFace datasets."""
    
    def __init__(self, embed_dim: int = EMBED_DIM, batch_size: int = BATCH_SIZE):
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer

    def preprocess_text(self, text: str) -> str:
        """
        Enhanced preprocessing for TF-IDF in RAG:
        - Lowercase
        - Remove non-alphabetic characters
        - Tokenize
        - Remove stopwords
        - Lemmatize
        - Rejoin to string
        """
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        if self.lemmatizer:
            tokens = nltk.word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
        else:
            tokens = [
                token for token in text.split() 
                if token not in self.stop_words and len(token) > 2
            ]
            
        return ' '.join(tokens)

    def stream_jsonl_text(self, jsonl_file: Union[str, Path]) -> Generator[str, None, None]:
        """Stream preprocessed text from a JSONL file."""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    yield self.preprocess_text(doc.get('text', ''))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue

    def stream_jsonl_documents(self, jsonl_file: Union[str, Path]) -> Generator[Tuple[str, str], None, None]:
        """Stream document ID and preprocessed text from a JSONL file."""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    doc_id = doc.get('doc_id', '')
                    text = doc.get('text', '')
                    yield doc_id, self.preprocess_text(text)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON line: {e}")
                    continue

    def load_vectorizer_from_config(self, model_config_file: Optional[str]) -> Optional[TfidfVectorizer]:
        """
        Load a pre-trained TF-IDF vectorizer from a model configuration file.
        Now loads from HuggingFace dataset instead of numpy files.
        """
        if not model_config_file or not os.path.exists(model_config_file):
            print("No model config file provided or file does not exist. Creating new vectorizer.")
            return None
            
        print(f"Loading pre-trained vectorizer from config file: {model_config_file}")
        
        try:
            with open(model_config_file, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading config file: {e}. Creating new vectorizer.")
            return None
        
        if 'vectorizer_dataset_path' not in config:
            print("Warning: 'vectorizer_dataset_path' not found in config file. Creating new vectorizer.")
            return None
            
        vectorizer_dataset_path = config['vectorizer_dataset_path']
        
        if not os.path.exists(vectorizer_dataset_path):
            print(f"Warning: Vectorizer dataset not found at {vectorizer_dataset_path}. Creating new vectorizer.")
            return None

        try:
            print(f"Loading vectorizer components from dataset: {vectorizer_dataset_path}")
            vectorizer_dataset = Dataset.load_from_disk(vectorizer_dataset_path)
            
            # Extract vocabulary and IDF values from dataset
            vocabulary_words = vectorizer_dataset['vocabulary']
            idf_values = np.array(vectorizer_dataset['idf_values'])
            
            # Create vocabulary dictionary
            vocab_dict = {word: idx for idx, word in enumerate(vocabulary_words)}
            
            # Initialize TF-IDF vectorizer with loaded vocabulary and IDF values
            print("Initializing TF-IDF vectorizer with loaded components...")
            tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_dict)
            tfidf_vectorizer.idf_ = idf_values
            
            print("Pre-trained vectorizer loaded successfully from HuggingFace dataset")
            return tfidf_vectorizer
            
        except Exception as e:
            print(f"Error loading vectorizer components from dataset: {e}. Creating new vectorizer.")
            return None

    def create_new_vectorizer(self, vectorizer_dataset_path: str, jsonl_file: Union[str, Path]) -> TfidfVectorizer:
        """
        Create a new TF-IDF vectorizer from the input documents.
        Now saves components to HuggingFace dataset instead of numpy files.
        """
        # Step 1: Build vocabulary  
        print("Building vocabulary...")
        vocab_vectorizer = CountVectorizer(max_features=self.embed_dim)
        vocab_vectorizer.fit(self.stream_jsonl_text(jsonl_file))

        # Step 2: Initialize TF-IDF vectorizer with fixed vocabulary
        print("Creating TF-IDF vectorizer...")
        tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_vectorizer.vocabulary_)
        tfidf_vectorizer.fit(self.stream_jsonl_text(jsonl_file))

        # Save vectorizer components to HuggingFace dataset
        print("Saving vectorizer components to HuggingFace dataset...")
        vocabulary_words = list(tfidf_vectorizer.vocabulary_.keys())
        idf_values = tfidf_vectorizer.idf_.tolist()
        
        # Create dataset with vectorizer components
        vectorizer_data = {
            'vocabulary': vocabulary_words,
            'idf_values': idf_values,
            'vocab_size': [len(vocabulary_words)] * len(vocabulary_words),
            'embedding_dim': [self.embed_dim] * len(vocabulary_words),
            'created_from': [str(jsonl_file)] * len(vocabulary_words)
        }
        
        vectorizer_dataset = Dataset.from_dict(vectorizer_data)
        vectorizer_dataset.save_to_disk(vectorizer_dataset_path)
        
        print(f"Vectorizer components saved to dataset: {vectorizer_dataset_path}")
        
        return tfidf_vectorizer

    def process_documents(
        self,
        jsonl_file: Union[str, Path],
        output_folder: Union[str, Path],
        chunk_size: int = -1,
        chunk_overlap: int = -1,
        model_config_file: Optional[str] = None,
        chunk_method: str = "fixed",
        dataset_name: str = DEFAULT_DATASET_NAME
    ) -> str:
        """
        Process a JSONL file containing documents and generate TF-IDF embeddings.
        Save as HuggingFace dataset.
        
        Returns:
            Path to the saved dataset directory
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        vectorizer_dataset_path = str(output_folder / "vectorizer_components")
        dataset_path = output_folder / dataset_name
        
        # Load or create vectorizer
        tfidf_vectorizer = self.load_vectorizer_from_config(model_config_file)
        if tfidf_vectorizer is None:
            tfidf_vectorizer = self.create_new_vectorizer(vectorizer_dataset_path, jsonl_file)
            
            # Create/update config file with dataset path
            config_data = {"vectorizer_dataset_path": vectorizer_dataset_path}
            config_file_path = output_folder / "vectorizer_config.json"
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Vectorizer config saved to: {config_file_path}")

        # Process documents in batches
        print("Processing documents and generating embeddings...")
        all_data = []
        
        chunk_iterator = chunk_text(chunk_method)
        
        for doc_id, text in self.stream_jsonl_documents(jsonl_file):
            chunk_index = 0
            for chunk in chunk_iterator(text, chunk_size, chunk_overlap):
                # Generate TF-IDF embedding for this chunk
                tfidf_vector = tfidf_vectorizer.transform([chunk]).toarray()[0]
                
                # Prepare data record
                record = {
                    "id": f"tfidf-{doc_id}-{chunk_index}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "embedding": tfidf_vector.tolist(),
                    "source": str(jsonl_file),
                    "embedding_model": "tfidf",
                    "embedding_dim": len(tfidf_vector)
                }
                
                all_data.append(record)
                chunk_index += 1
                
                # Process in batches to manage memory
                if len(all_data) >= self.batch_size:
                    self._save_batch_to_dataset(all_data, dataset_path, append=len(all_data) > self.batch_size)
                    print(f"Processed batch of {len(all_data)} chunks")
                    all_data = []

        # Process any remaining data
        if all_data:
            self._save_batch_to_dataset(all_data, dataset_path, append=True)
            print(f"Processed final batch of {len(all_data)} chunks")

        print(f"Dataset saved to: {dataset_path}")
        return str(dataset_path)

    def _save_batch_to_dataset(self, data: List[Dict], dataset_path: Path, append: bool = False):
        """Save a batch of data to HuggingFace dataset."""
        dataset = Dataset.from_list(data)
        
        if append and dataset_path.exists():
            # Load existing dataset and concatenate
            try:
                existing_dataset = Dataset.load_from_disk(str(dataset_path))
                from datasets import concatenate_datasets
                dataset = concatenate_datasets([existing_dataset, dataset])
            except Exception as e:
                print(f"Warning: Could not append to existing dataset: {e}")
        
        dataset.save_to_disk(str(dataset_path))


# Legacy function for backward compatibility
def process_jsonl_documents_tfidf(
    jsonl_file: str,
    output_folder: str,
    chunk_size: int = -1,
    chunk_overlap: int = -1,
    model_config_file: Optional[str] = None,
    chunk_method: str = "fixed"
) -> str:
    """
    Legacy wrapper function for backward compatibility.
    Now saves as HuggingFace dataset instead of JSONL.
    """
    processor = TfidfEmbeddingProcessor()
    
    # Convert output_file to directory path for dataset
    if output_folder.endswith('/'):
        output_path = output_folder
    else:
        output_path = f"{output_folder}/"
    
    return processor.process_documents(
        jsonl_file=jsonl_file,
        output_folder=output_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_config_file=model_config_file,
        chunk_method=chunk_method
    )


# Remove unused legacy functions
# stream_document_batches, stream_all_documents, load_vectorizer_from_config, create_new_vectorizer
# are now methods of TfidfEmbeddingProcessor class
