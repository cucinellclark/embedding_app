import requests
import json
import os
import glob
import re
import sys

def chunk_text(text, chunk_size, chunk_overlap):
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


def embed_document(api_key, endpoint, model_name, text, doc_id, source, output, chunk_size=-1, chunk_overlap=-1):
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
    """
    # Define headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Split text into chunks if chunk_size is specified
    chunk_index = 0
    for chunk in chunk_text(text, chunk_size, chunk_overlap):
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
