"""
LLM-based document extraction using OpenAI GPT
"""
import os
from typing import Dict, Any, List
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMExtractor:
    """Extract structured information from documents using GPT"""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM extractor with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. Set it in .env file or pass as parameter.")
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "gpt-4o-mini"  # Cost-effective model for extraction
        
    def extract(self, text: str, rag_context: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract structured information from document text
        
        Args:
            text: OCR-extracted text from document
            rag_context: Optional context from RAG system
            
        Returns:
            Dictionary with extracted fields
        """
        if not self.client:
            return self._fallback_extraction(text)
        
        try:
            # Build context from RAG if available
            context_info = ""
            if rag_context:
                context_info = "\n\nSimilar documents for reference:\n"
                for i, ctx in enumerate(rag_context[:3]):  # Use top 3 similar docs
                    labels = ctx.get('labels', {})
                    context_info += f"\nDocument {i+1}: {json.dumps(labels, indent=2)}\n"
            
            # Create extraction prompt
            prompt = self._create_extraction_prompt(text, context_info)
            
            # Call GPT for extraction
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document extraction system. Extract structured information from documents accurately."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"}
            )
            
            # Parse response
            extracted_data = json.loads(response.choices[0].message.content)
            
            return {
                'llm_results': extracted_data,
                'rag_context': rag_context or [],
                'enhanced_results': extracted_data,
                'success': True,
                'message': 'LLM extraction completed successfully'
            }
            
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self._fallback_extraction(text)
    
    def _create_extraction_prompt(self, text: str, context: str = "") -> str:
        """Create extraction prompt for GPT"""
        prompt = f"""
Extract structured information from the following document text.

Document Text:
{text}

{context}

Extract the following fields if present in the document:
- document_type: Type of document (invoice, receipt, contract, form, etc.)
- date: Any dates mentioned
- amount: Any monetary amounts
- names: Any person or company names
- addresses: Any addresses
- phone_numbers: Any phone numbers
- email_addresses: Any email addresses
- reference_numbers: Any reference or ID numbers
- total_amount: Total amount if applicable
- items: List of items with quantities and prices if applicable
- other_fields: Any other important information

Return the result as a JSON object with the extracted fields. If a field is not found, set it to null.
"""
        return prompt
    
    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback extraction when LLM is not available"""
        # Simple rule-based extraction as fallback
        import re
        
        results = {
            'document_type': 'unknown',
            'date': None,
            'amount': None,
            'names': [],
            'addresses': [],
            'phone_numbers': [],
            'email_addresses': [],
            'reference_numbers': [],
            'total_amount': None,
            'items': [],
            'other_fields': {}
        }
        
        # Extract dates
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}\.\d{2}\.\d{4}'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                results['date'] = matches[0]
                break
        
        # Extract amounts
        amount_pattern = r'[$€£]?\s*[\d,]+\.?\d*\s*(?:USD|EUR|GBP)?'
        amounts = re.findall(amount_pattern, text)
        if amounts:
            results['amount'] = amounts[0]
            results['total_amount'] = amounts[-1]  # Last amount often total
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        results['email_addresses'] = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_pattern = r'\+?[\d\s\-\(\)]{10,}'
        results['phone_numbers'] = re.findall(phone_pattern, text)
        
        return {
            'llm_results': results,
            'rag_context': [],
            'enhanced_results': results,
            'success': True,
            'message': 'Fallback extraction completed (LLM not available)'
        }

if __name__ == "__main__":
    # Test the LLM extractor
    extractor = LLMExtractor()
    
    test_text = """
    INVOICE #12345
    Date: 2024-01-15
    From: ABC Company
    Email: contact@abc.com
    Phone: +1-555-123-4567
    
    Item 1: Widget A - $10.00
    Item 2: Widget B - $20.00
    
    Total: $30.00
    """
    
    results = extractor.extract(test_text)
    print("Extraction Results:")
    print(json.dumps(results, indent=2))
