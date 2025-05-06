import requests
import json
import os
import glob
import re
import sys
import nltk
from nltk.tokenize import sent_tokenize

def chunk_text_fixed(text, chunk_size, chunk_overlap):
    """
    Yield text chunks of specified size with overlap

    Args:
        text: Text to split into chunks
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Yields:
        Chunks of text
    """
    if chunk_size == -1:
        yield text
        return

    text_length = len(text)
    
    # Calculate the effective step size between chunks
    step_size = chunk_size - chunk_overlap if chunk_overlap > 0 else chunk_size
    print(f"step_size: {step_size}")
    
    # Calculate the number of chunks
    if step_size > 0:
        num_chunks = (text_length + step_size - 1) // step_size
    else:
        num_chunks = 1
    print(f"num_chunks: {num_chunks}")
    
    for i in range(num_chunks):
        start = i * step_size
        end = min(start + chunk_size, text_length)
        yield text[start:end]
        
        # Break if we've reached the end of the text
        if end >= text_length:
            break

def chunk_text_sentence(text, chunk_size, chunk_overlap):
    """
    Yield text chunks based on sentences while trying to maintain approximate chunk size and overlap.
    Uses a more robust sentence splitting approach that can handle missing punctuation.

    Args:
        text: Text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Target overlap between chunks in characters

    Yields:
        Chunks of text
    """
    if chunk_size == -1:
        yield text
        return

    # First try NLTK's sentence tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
        sentences = sent_tokenize(text)
    except (LookupError, Exception):
        # Fallback to a more robust splitting approach
        # Split on common sentence endings, but also handle cases where they might be missing
        sentences = []
        current_sentence = []
        words = text.split()
        
        for word in words:
            current_sentence.append(word)
            # Check for sentence endings or if we've reached a reasonable length
            if (word.endswith(('.', '!', '?')) or 
                len(' '.join(current_sentence)) > 100):  # Arbitrary length threshold
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        
        # Add any remaining text as a sentence
        if current_sentence:
            sentences.append(' '.join(current_sentence))

    current_chunk = []
    current_size = 0
    overlap_buffer = []
    overlap_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed chunk_size, yield current chunk
        if current_size + sentence_size > chunk_size and current_chunk:
            yield ' '.join(current_chunk)
            
            # Prepare overlap buffer
            overlap_buffer = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= chunk_overlap:
                    overlap_buffer.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk = overlap_buffer
            current_size = overlap_size
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_size += sentence_size

    # Yield the last chunk if it exists
    if current_chunk:
        yield ' '.join(current_chunk)

def chunk_text_words(text, chunk_size, chunk_overlap):
    """
    Yield text chunks based on approximate word count while trying to maintain chunk size and overlap.

    Args:
        text: Text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Target overlap between chunks in characters

    Yields:
        Chunks of text
    """
    if chunk_size == -1:
        yield text
        return

    # Split text into words
    words = text.split()
    if not words:
        return

    # Calculate approximate words per chunk based on average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    words_per_chunk = max(1, int(chunk_size / avg_word_length))
    words_overlap = max(1, int(chunk_overlap / avg_word_length))

    current_chunk = []
    current_size = 0
    overlap_buffer = []

    for word in words:
        word_size = len(word)
        
        # If adding this word would exceed chunk_size, yield current chunk
        if current_size + word_size > chunk_size and current_chunk:
            yield ' '.join(current_chunk)
            
            # Prepare overlap buffer
            overlap_buffer = current_chunk[-words_overlap:] if words_overlap < len(current_chunk) else current_chunk
            
            # Start new chunk with overlap
            current_chunk = overlap_buffer
            current_size = sum(len(w) for w in current_chunk) + len(current_chunk) - 1  # -1 for spaces
        
        # Add word to current chunk
        current_chunk.append(word)
        current_size += word_size + 1  # +1 for space

    # Yield the last chunk if it exists
    if current_chunk:
        yield ' '.join(current_chunk)

def chunk_text(method):
    """
    Return the appropriate chunking function based on the specified method.
    
    Args:
        method: The chunking method to use ("fixed", "sentence", or "words")
        
    Returns:
        The appropriate chunking function
        
    Raises:
        ValueError: If an invalid chunking method is specified
    """
    chunking_functions = {
        "fixed": chunk_text_fixed,
        "sentence": chunk_text_sentence,
        "words": chunk_text_words
    }
    
    if method not in chunking_functions:
        raise ValueError(f"Invalid chunk_method: {method}. Must be one of {list(chunking_functions.keys())}")
    
    return chunking_functions[method]

def embed_document(api_key, endpoint, model_name, text, doc_id, source, output, chunk_size=-1, chunk_overlap=-1, chunk_method="fixed"):
    """
    Embed a text content using the specified API.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        text: Text content to embed
        source: Source of the text (e.g., file path)
        output: Path to the output file
        chunk_size: Size of text chunks (if applicable)
        chunk_overlap: Overlap between chunks (if applicable)
        chunk_method: Method to use for chunking ("fixed", "sentence", or "words")
    """
    # Define headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Get the appropriate chunking function
    chunk_iterator = chunk_text(chunk_method)

    # Split text into chunks using the selected method
    chunk_index = 0
    for chunk in chunk_iterator(text, chunk_size, chunk_overlap):
        payload = {
            "model": model_name,
            "input": chunk 
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            
            # Check response
            if response.status_code == 200:
                print(f"Connection successful for chunk {chunk_index}!")
                res_json = response.json()
                data = res_json['data'][0]  # [{'id','object','embedding'},{'id','object','embedding'}]
                data['source'] = source
                data['doc_id'] = doc_id
                data['chunk_index'] = chunk_index
                data['chunk_text'] = chunk
                
                with open(output, 'a') as o:
                    text = json.dumps(data)
                    o.write(text)
                    o.write('\n')
            else:
                print(f"Failed to connect for chunk {chunk_index}: {response.status_code}")
                print("Response:", response.text)
                with open(output, 'a') as o:
                    error_data = {
                        "status": "failed",
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "chunk_text": chunk,
                        "error_code": response.status_code,
                        "error_message": response.text
                    }
                    o.write(json.dumps(error_data))
                    o.write('\n')

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to {endpoint} for chunk {chunk_index}: {e}")
            with open(output, 'a') as o:
                error_data = {
                    "status": "failed",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "error_type": "RequestException",
                    "error_message": str(e)
                }
                o.write(json.dumps(error_data))
                o.write('\n')
        finally:
            chunk_index += 1

def test_embedding_endpoint(api_key, endpoint, model_name):
    """
    Test the embedding endpoint to ensure it's working properly.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        
    Returns:
        bool: True if the endpoint is working, False otherwise
    """
    print(f"Testing embedding endpoint: {endpoint}")
    
    # Define headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create a simple test payload
    payload = {
        "model": model_name,
        "input": "This is a test request to verify the embedding endpoint is working."
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        
        # Check response
        if response.status_code == 200:
            res_json = response.json()
            print("✅ Embedding endpoint test successful!")
            return True
        else:
            print(f"❌ Embedding endpoint test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to embedding endpoint: {e}")
        return False
