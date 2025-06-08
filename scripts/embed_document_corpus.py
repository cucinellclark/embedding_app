import requests
import json
import os,glob,sys
import argparse
from embedding_utils import embed_document, test_embedding_endpoint
from text_utils import validate_jsonl_file, validate_jsonl_files_in_directory
from tfidf_embed import process_jsonl_documents_tfidf

def process_document_file(api_key, endpoint, model_name, document_file, output_folder, chunk_size=-1, chunk_overlap=-1, model_config_file=None, chunk_method="fixed"):

    """
    Process a single document file for embedding.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        document_file: Path to the document file
        output_folder: Path to the output folder where embeddings will be saved
        chunk_size: Size of text chunks (if applicable)
        chunk_overlap: Overlap between chunks (if applicable)
        model_config_file: Path to a JSON file containing model configuration (for TF-IDF)
        chunk_method: Method to use for chunking text
    """
    
    print(f"Processing document file: {document_file}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate output file name based on input file name
    base_name = os.path.splitext(os.path.basename(document_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}_embeddings.jsonl")
    
    # Handle TF-IDF model
    if model_name.lower() == "tfidf":
        print("Using TF-IDF embedding model")
        output_file = os.path.join(output_folder, f"{base_name}_embeddings.jsonl")
        dataset_path = process_jsonl_documents_tfidf(document_file, output_file, chunk_size, chunk_overlap, model_config_file, chunk_method)
        print(f"TF-IDF embeddings dataset saved to: {dataset_path}")
        return
    
    # Regular embedding process
    with open(document_file, 'r') as f:
        for line in f:
            file_json = json.loads(line)
            text = file_json['text']
            doc_id = file_json['doc_id']
            embed_document(api_key, endpoint, model_name, text, doc_id, document_file, output_file, chunk_size, chunk_overlap, chunk_method)
    
    print(f"Embeddings saved to: {output_file}")

def process_document_folder(api_key, endpoint, model_name, document_folder, output_folder, chunk_size=-1, chunk_overlap=-1, model_config_file=None, chunk_method="fixed"):
    """
    Process all documents in a folder for embedding.
    
    Args:
        api_key: API key for authentication
        endpoint: API endpoint URL
        model_name: Name of the model to use
        document_folder: Path to the folder containing documents
        output_folder: Path to the output folder where embeddings will be saved
        chunk_size: Size of text chunks (if applicable)
        chunk_overlap: Overlap between chunks (if applicable)
        model_config_file: Path to a JSON file containing model configuration (for TF-IDF)
        chunk_method: Method to use for chunking text
    """
    print(f"Processing documents in folder: {document_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
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
        
        # Use combined output file for all documents in folder
        combined_output_file = os.path.join(output_folder, "combined_embeddings.jsonl")
        process_jsonl_documents_tfidf(temp_file, combined_output_file)
        
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
            
            # Generate output file name based on input file name
            base_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(output_folder, f"{base_name}_embeddings.jsonl")
            
            with open(file, 'r') as f:
                for line in f:
                    file_json = json.loads(line)
                    text = file_json['text']
                    doc_id = file_json['doc_id']
                    embed_document(api_key, endpoint, model_name, text, doc_id, file, output_file, chunk_size, chunk_overlap, chunk_method)
            
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
    parser.add_argument("--output_folder", type=str, required=True)
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
    output_folder = args.output_folder
    terminate_on_error = args.terminate_on_error
    model_config_file = args.model_config_file
    # Check output parameters first
    if not output_folder:
        print("Error: output_folder is required.")
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
        process_document_file(api_key, endpoint, model_name, document_file, output_folder, chunk_size, chunk_overlap, model_config_file)
    elif document_folder: 
        process_document_folder(api_key, endpoint, model_name, document_folder, output_folder, chunk_size, chunk_overlap, model_config_file, chunk_method)
    else:
        print("Error: Please provide either document_file or document_folder.")
        sys.exit(1)
