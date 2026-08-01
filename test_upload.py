"""
Test script for document upload with readable output
"""
import requests
import json
import pandas as pd
from pathlib import Path

def test_document_upload(file_path):
    """Test document upload and display results in readable format"""
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return
    
    # Upload file
    print(f"Uploading file: {file_path}")
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/extract-file', files=files)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    # Parse response
    result = response.json()
    
    print("\n" + "="*80)
    print("DOCUMENT EXTRACTION RESULTS")
    print("="*80)
    
    # Display Extraction Results (LLM only)
    print("\n📊 EXTRACTION RESULTS")
    print("-" * 40)
    extraction_results = result.get('extraction_results', {})
    for key, value in extraction_results.items():
        # Handle different value types for clean display
        if isinstance(value, list):
            if value:
                print(f"{key:30} : {value}")
            else:
                print(f"{key:30} : []")
        elif isinstance(value, dict):
            if value:
                print(f"{key:30} : {value}")
            else:
                print(f"{key:30} : {{}}")
        else:
            print(f"{key:30} : {value}")
    
    # Display RAG Context
    print("\n🔍 RAG CONTEXT (Similar Documents)")
    print("-" * 40)
    rag_context = result.get('rag_context', [])
    for i, context in enumerate(rag_context):
        print(f"\nDocument {i+1}:")
        print(f"  Distance: {context.get('distance', 'N/A'):.4f}")
        print(f"  Source: {context.get('metadata', {}).get('source_dataset', 'N/A')}")
        print(f"  Type: {context.get('metadata', {}).get('doc_type', 'N/A')}")
        labels = context.get('labels', {})
        print(f"  Fields: {', '.join(list(labels.keys())[:5])}")
    
    # Create DataFrame for Excel-like view
    print("\n📋 EXCEL-LIKE VIEW")
    print("-" * 40)
    
    # Build table from extraction results
    table_data = []
    for key, value in extraction_results.items():
        # Convert complex types to strings for CSV
        if isinstance(value, (list, dict)):
            value_str = str(value)
        else:
            value_str = str(value) if value is not None else 'N/A'
        
        row = {
            'Field': key,
            'Value': value_str
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = "extraction_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print(f"Status: {result.get('success', False)}")
    print(f"Message: {result.get('message', 'N/A')}")
    print("="*80)

if __name__ == "__main__":
    # Get file path from user
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default test file - you can change this
        file_path = input("Enter path to document image: ")
    
    test_document_upload(file_path)
