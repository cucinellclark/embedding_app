import requests
import json
import os,glob,sys
import argparse
from embedding_utils import embed_document, test_embedding_endpoint
from text_utils import validate_jsonl_file, validate_jsonl_files_in_directory
from tfidf_embed import process_jsonl_documents_tfidf

def process_document_file(api_key, endpoint, model_name, document_file, output_file, chunk_size=-1, chunk_overlap=-1, model_config_file=None, chunk_method="fixed"):

    """
    Process a single document file for embedding.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        document_file: Path to the document file
        output_file: Path to the output file
        chunk_size: Size of text chunks (if applicable)
        chunk_overlap: Overlap between chunks (if applicable)
        vectorizer_file: Base path for saving vectorizer components (for TF-IDF)
        embed_dim: Dimension of the embedding vectors (for TF-IDF)
    """
    
    print(f"Processing document file: {document_file}")
    
    # Handle TF-IDF model
    # if there is not a model config file, then we are training a new model
    # if there is a model config file, then we are using a pre-trained model
    if model_name.lower() == "tfidf":
        print("Using TF-IDF embedding model")
        process_jsonl_documents_tfidf(document_file, output_file, chunk_size, chunk_overlap, chunk_method, model_config_file)
        return
    
    # Regular embedding process
    with open(document_file, 'r') as f:
        for line in f:
            file_json = json.loads(line)
            text = file_json['text']
            doc_id = file_json['doc_id']
            embed_document(api_key, endpoint, model_name, text, doc_id, document_file, output_file, chunk_size, chunk_overlap, chunk_method)
    
    print(f"Embeddings saved to: {output_file}")

def process_document_folder(api_key, endpoint, model_name, document_folder, output_file, chunk_size=-1, chunk_overlap=-1, model_config_file=None):
    """
    Process all documents in a folder for embedding.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        document_folder: Path to the folder containing documents
        output_file: Path to the output file
        chunk_size: Size of text chunks (if applicable)
        chunk_overlap: Overlap between chunks (if applicable)
        vectorizer_file: Base path for saving vectorizer components (for TF-IDF)
        embed_dim: Dimension of the embedding vectors (for TF-IDF)
    """
    print(f"Processing documents in folder: {document_folder}")
    
    # Handle TF-IDF model
    # if there is not a model config file, then we are training a new model
    # if there is a model config file, then we are using a pre-trained model
    if model_name.lower() == "tfidf":
        print('haven\'t implemented this yet')
        sys.exit(1)
        print("Using TF-IDF embedding model")
        
        # Create a temporary file listing all documents in the folder
        temp_file = "temp_doc_list.txt"
        with open(temp_file, 'w') as f:
            for file in glob.glob(os.path.join(document_folder, "*")):
                if os.path.isfile(file):
                    f.write(f"{file}\n")
        
        process_jsonl_documents_tfidf(temp_file, output_file)
        
        # Clean up temporary file
        os.remove(temp_file)
        return
    
    # Regular embedding process
    # Get all files in the folder
    files = glob.glob(os.path.join(document_folder, "*"))
    
    # Process each file
    for file in files:
        if os.path.isfile(file):
            print(f"Processing file: {file}")
            
            with open(file, 'r') as f:
                for line in f:
                    file_json = json.loads(line)
                    text = file_json['text']
                    doc_id = file_json['doc_id']
                    embed_document(api_key, endpoint, model_name, text, doc_id, file, output_file, chunk_size, chunk_overlap)
            
            print(f"Embeddings saved to: {output_file}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=False, default="EMPTY")
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--document_file", type=str, required=False, default=None)
    parser.add_argument("--document_folder", type=str, required=False, default=None)
    parser.add_argument("--chunk_size", type=int, required=False, default=-1)
    parser.add_argument("--chunk_overlap", type=int, required=False, default=-1)
    parser.add_argument("--chunk_method", type=str, required=False, default="fixed")
    parser.add_argument("--terminate_on_error", action="store_true", required=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_config_file", type=str, required=False, default=None)
    args = parser.parse_args()

    api_key = args.api_key
    endpoint = args.endpoint
    model_name = args.model_name
    document_file = args.document_file
    document_folder = args.document_folder
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    chunk_method = args.chunk_method
    output_file = args.output_file
    terminate_on_error = args.terminate_on_error
    model_config_file = args.model_config_file
    # Check output parameters first
    if not output_file:
        print("Error: output_file is required.")
        sys.exit(1)

    # Validate input files before testing the embedding endpoint
    if document_file:
        print(f"Validating document file: {document_file}")
        success, errors = validate_jsonl_file(document_file)
        if not success:
            print("Document validation failed:")
            for error in errors:
                print(f"  - {error}")
        if terminate_on_error:
            print("Document validation failed. Exiting.")
            sys.exit(1)
        print("Document validation successful.")
    
    elif document_folder:
        print(f"Validating documents in folder: {document_folder}")
        validation_results = validate_jsonl_files_in_directory(document_folder)
        validation_failed = False
        
        for file_path, (success, errors) in validation_results.items():
            if not success:
                validation_failed = True
                print(f"Validation failed for {file_path}:")
                for error in errors:
                    print(f"  - {error}")
        
        if validation_failed and terminate_on_error:
            print("Document validation failed. Exiting.")
            sys.exit(1)
        print("All documents validated successfully.")
    
    # Test the embedding endpoint before processing any documents
    # Skip endpoint test for TF-IDF model
    if model_name.lower() != "tfidf" and not test_embedding_endpoint(api_key, endpoint, model_name):
        print("Embedding endpoint test failed. Exiting program. Check error log for more details.")
        sys.exit(1)

    # Process either document file or document folder, but not both
    if document_file and document_folder:
        print("Error: Please provide either document_file or document_folder, not both.")
        sys.exit(1)
    elif document_file:
        process_document_file(api_key, endpoint, model_name, document_file, output_file, chunk_size, chunk_overlap, model_config_file)
    elif document_folder: 
        process_document_folder(api_key, endpoint, model_name, document_folder, output_file, chunk_size, chunk_overlap, model_config_file)
    else:
        print("Error: Please provide either document_file or document_folder.")
        sys.exit(1)
