# Embedding Application

## Overview

This repository holds a BV-BRC module.

## About this module

This module is a component of the BV-BRC build system. It is designed to fit into the
`dev_container` infrastructure which manages development and production deployment of
the components of the BV-BRC. More documentation is available [here](https://github.com/BV-BRC/dev_container/tree/master/README.md).

This is hard coded to work with the following response schema:
<pre>
{
    'id': vllm-server-id,
    'object': 'list',
    'created': created_date
    'model': model_name,
    'data': [
        {
            'index': 0,
            'object': 'embedding',
            'embedding': [...],
            'usage': { tokenization_data }
        }
    ]
}
</pre>

Outputs the embeddings in a jsonl file like so:
<pre>
{
    "index": 0,
    "object": "embedding",
    "source": source file,
    "doc_id": unique generated id linking back to full document,
    "chunk_index": index of the chunk of the full document,
    "chunk_text": the embedding text of this chunk (could be the full document)
    "embedding": [...]
}
</pre>

Will accept two formats for input: a folder or a jsonl file

Below is the expected format for the file input, along with each file in the folder input option
<pre>
{"doc_id": "unique_id_1", "text": "This is the document text I would like to embed...", "metadata": [ ... ]}
{"doc_id": "unique_id_2", "text": "This is the document text I would like to embed that is different...", "metadata": [ ... ]}
</pre>

## TF-IDF Embeddings with HuggingFace Datasets

The TF-IDF embedding functionality has been modernized to use HuggingFace datasets with PyArrow for efficient storage and retrieval. This provides several benefits:

- **Efficient Storage**: PyArrow provides columnar storage that's more efficient than JSONL
- **Easy Loading**: Datasets can be easily loaded and manipulated with the `datasets` library
- **Better Memory Management**: Support for memory mapping and streaming large datasets
- **Compatibility**: Works seamlessly with other HuggingFace tools and PyTorch/TensorFlow

### Usage

#### Basic Usage with TfidfEmbeddingProcessor Class

```python
from lib.tfidf_embed import TfidfEmbeddingProcessor
from datasets import Dataset

# Initialize processor
processor = TfidfEmbeddingProcessor(embed_dim=768, batch_size=100)

# Process documents and save as HuggingFace dataset
dataset_path = processor.process_documents(
    jsonl_file="input_documents.jsonl",
    output_path="embeddings_output",
    chunk_size=1000,
    chunk_overlap=100,
    chunk_method="sentence",
    dataset_name="my_embeddings"
)

# Load the dataset
dataset = Dataset.load_from_disk(dataset_path)
print(f"Dataset contains {len(dataset)} embeddings")
```

#### Legacy Function (Backward Compatibility)

The original `process_jsonl_documents_tfidf` function is still available and now saves data as HuggingFace datasets:

```python
from lib.tfidf_embed import process_jsonl_documents_tfidf

# This now returns the path to a HuggingFace dataset directory
dataset_path = process_jsonl_documents_tfidf(
    jsonl_file="input.jsonl",
    output_file="output.jsonl",  # Will be converted to output_dataset/
    chunk_size=1000,
    chunk_overlap=100,
    model_config_file=None,
    chunk_method="fixed"
)
```

### Dataset Schema

The generated datasets have the following schema:

```python
{
    "id": str,              # Unique identifier: "tfidf-{doc_id}-{chunk_index}"
    "doc_id": str,          # Original document ID
    "chunk_index": int,     # Index of the chunk within the document
    "text": str,            # The actual text chunk
    "embedding": List[float], # TF-IDF vector as list of floats
    "source": str,          # Path to source JSONL file
    "embedding_model": str, # Always "tfidf"
    "embedding_dim": int    # Dimension of the embedding vector
}
```

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `datasets>=2.0.0` - HuggingFace datasets library
- `pyarrow>=10.0.0` - Efficient columnar storage
- `scikit-learn>=1.0.0` - TF-IDF vectorization
- `nltk>=3.6.0` - Text preprocessing

### Example

See `example_tfidf_usage.py` for a complete working example of the new functionality.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if not already present)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
