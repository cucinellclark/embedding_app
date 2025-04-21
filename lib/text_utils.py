import json
import os
from typing import List, Dict, Tuple

def validate_jsonl_file(file_path: str, required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate a JSONL file by checking if all required fields exist in each JSON object.
    
    Args:
        file_path: Path to the JSONL file
        required_fields: List of required field names. Defaults to ['text', 'id'] if None.
        
    Returns:
        Tuple containing:
            - Boolean indicating if validation passed
            - List of error messages (empty if validation passed)
    """
    if required_fields is None:
        required_fields = ['text', 'doc_id']
    
    errors = []
    
    if not os.path.exists(file_path):
        return False, [f"File not found: {file_path}"]
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Check if all required fields exist
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        errors.append(f"Line {line_num}: Missing required fields: {', '.join(missing_fields)}")
                    
                    # Check if fields have non-empty values
                    for field in required_fields:
                        if field in data and (data[field] is None or data[field] == ""):
                            errors.append(f"Line {line_num}: Field '{field}' has an empty value")
                
                except json.JSONDecodeError:
                    errors.append(f"Line {line_num}: Invalid JSON format")
    
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
    
    return len(errors) == 0, errors

def validate_jsonl_files_in_directory(directory_path: str, required_fields: List[str] = None) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate all JSONL files in a directory.
    
    Args:
        directory_path: Path to the directory containing JSONL files
        required_fields: List of required field names. Defaults to ['text', 'id'] if None.
        
    Returns:
        Dictionary mapping file paths to validation results (success boolean and error messages)
    """
    results = {}
    
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {directory_path: (False, ["Directory not found or is not a directory"])}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl') or filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            results[file_path] = validate_jsonl_file(file_path, required_fields)
    
    return results
