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
        # Enhanced rule-based extraction as fallback
        import re
        
        results = {
            'document_type': self._detect_document_type(text),
            'date': self._extract_dates(text),
            'amount': self._extract_amounts(text),
            'names': self._extract_names(text),
            'addresses': self._extract_addresses(text),
            'phone_numbers': self._extract_phone_numbers(text),
            'email_addresses': self._extract_emails(text),
            'reference_numbers': self._extract_reference_numbers(text),
            'total_amount': self._extract_total_amount(text),
            'items': self._extract_items(text),
            'other_fields': self._extract_other_fields(text)
        }
        
        return {
            'llm_results': results,
            'rag_context': [],
            'enhanced_results': results,
            'success': True,
            'message': 'Fallback extraction completed (LLM not available)'
        }
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type from text"""
        text_lower = text.lower()
        
        # Check for chat/conversation first
        if self._is_chat_conversation(text):
            return 'chat_conversation'
        
        type_keywords = {
            'invoice': ['invoice', 'bill', 'rechnung', 'fatura'],
            'receipt': ['receipt', 'quittung', 'fiş', 'receipt'],
            'contract': ['contract', 'agreement', 'vertrag', 'sözleşme'],
            'form': ['form', 'application', 'antrag', 'başvuru'],
            'letter': ['letter', 'brief', 'mektup'],
            'report': ['report', 'bericht', 'rapor'],
            'certificate': ['certificate', 'zertifikat', 'sertifika'],
            'statement': ['statement', 'kontoauszug', 'ekstre']
        }
        
        for doc_type, keywords in type_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        
        return 'unknown'
    
    def _is_chat_conversation(self, text: str) -> bool:
        """Detect if text is a chat conversation"""
        import re
        
        # Chat indicators
        chat_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',  # Timestamps
            r'\d{1,2}\.\d{2}\s*(?:AM|PM|am|pm)?',  # European timestamps
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Date patterns in chat
            r'w\s*$',  # WhatsApp indicators
            r'\[.*?\]\s*:',  # Brackets with colons
            r'^[A-Za-z]+:\s*',  # Name: pattern at start
            r'^\d+:\s*\w+',  # Number: word pattern
        ]
        
        # Count matches
        chat_score = 0
        for pattern in chat_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            chat_score += len(matches)
        
        # If many chat patterns found, it's likely a conversation
        return chat_score > 5
    
    def _extract_dates(self, text: str) -> str:
        """Extract dates with multiple formats"""
        import re
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_amounts(self, text: str) -> str:
        """Extract monetary amounts with various formats"""
        import re
        
        # Multiple currency patterns
        amount_patterns = [
            r'[$€£₺]\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*(?:USD|EUR|GBP|TL|TRY)\b',
            r'[\d,]+\.?\d*\s*(?:Dollar|Euro|Pound|Lira)\b',
            r'\b\d+[.,]\d{2}\b'  # Numbers with 2 decimals
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_names(self, text: str) -> list:
        """Extract person and company names"""
        import re
        
        names = []
        
        # Company name patterns (capitalized words)
        company_patterns = [
            r'\b(?:Inc|Corp|LLC|Ltd|GmbH|AG|Co|Company)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|GmbH|AG|Co)\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        # Person name patterns (Title + Name)
        person_patterns = [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        return list(set(names))  # Remove duplicates
    
    def _extract_addresses(self, text: str) -> list:
        """Extract addresses"""
        import re
        
        addresses = []
        
        # Address patterns (street + number + city)
        address_patterns = [
            r'\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*,?\s*[A-Za-z]+(?:\s+[A-Za-z]+)*',
            r'[A-Za-z]+(?:\s+[A-Za-z]+)*\s+\d+,?\s*[A-Za-z]+(?:\s+[A-Za-z]+)*'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text)
            addresses.extend(matches)
        
        return list(set(addresses))
    
    def _extract_phone_numbers(self, text: str) -> list:
        """Extract phone numbers with various formats"""
        import re
        
        phone_patterns = [
            r'\+?[\d\s\-\(\)]{10,}',
            r'\d{3}[-\s]?\d{3}[-\s]?\d{4}',
            r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',
            r'\d{2,4}[-\s]?\d{2,4}[-\s]?\d{6,8}'
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        return list(set(phones))
    
    def _extract_emails(self, text: str) -> list:
        """Extract email addresses"""
        import re
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text, re.IGNORECASE)
    
    def _extract_reference_numbers(self, text: str) -> list:
        """Extract reference numbers, IDs, invoice numbers"""
        import re
        
        ref_patterns = [
            r'(?:invoice|order|reference|ref|id|number|#)\s*[:#]?\s*[A-Z0-9-]+',
            r'\b[A-Z]{2,4}-\d{4,10}\b',  # Pattern like INV-12345
            r'\b\d{6,}\b'  # Long numbers
        ]
        
        refs = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            refs.extend(matches)
        
        return list(set(refs))
    
    def _extract_total_amount(self, text: str) -> str:
        """Extract total amount specifically"""
        import re
        
        total_patterns = [
            r'(?:total|sum|gesamt|toplam)\s*[:#]?\s*[$€£₺]?\s*[\d,]+\.?\d*',
            r'(?:total|sum|gesamt|toplam)\s*[:#]?\s*[\d,]+\.?\d*\s*(?:USD|EUR|GBP|TL|TRY)\b'
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract just the amount part
                amount_match = re.search(r'[\d,]+\.?\d*', matches[0])
                if amount_match:
                    return amount_match.group()
        
        return None
    
    def _extract_items(self, text: str) -> list:
        """Extract line items from documents"""
        import re
        
        items = []
        
        # Item patterns (description + price)
        item_patterns = [
            r'[A-Za-z][A-Za-z0-9\s]+\s*[:#]?\s*[$€£₺]?\s*[\d,]+\.?\d*',
            r'\d+\s*[xX]?\s*[A-Za-z][A-Za-z0-9\s]+\s*[$€£₺]?\s*[\d,]+\.?\d*'
        ]
        
        for pattern in item_patterns:
            matches = re.findall(pattern, text)
            items.extend(matches)
        
        return items[:10]  # Limit to first 10 items
    
    def _extract_other_fields(self, text: str) -> dict:
        """Extract other important fields"""
        import re
        
        other_fields = {}
        
        # Check if it's a chat conversation and extract chat-specific fields
        if self._is_chat_conversation(text):
            other_fields.update(self._extract_chat_fields(text))
        else:
            # Extract key-value pairs for regular documents
            kv_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\s*[:#]\s*([^\n]+)'
            matches = re.findall(kv_pattern, text)
            
            for key, value in matches:
                if len(key) > 2 and len(value) > 0:  # Filter out noise
                    other_fields[key] = value.strip()
        
        return other_fields
    
    def _extract_chat_fields(self, text: str) -> dict:
        """Extract fields from chat conversations"""
        import re
        
        chat_fields = {
            'participants': self._extract_chat_participants(text),
            'messages': self._extract_chat_messages(text),
            'timestamps': self._extract_chat_timestamps(text),
            'message_count': len(re.findall(r'\d+:\d+', text))
        }
        
        return chat_fields
    
    def _extract_chat_participants(self, text: str) -> list:
        """Extract chat participants"""
        import re
        
        # Pattern for names before messages
        participant_patterns = [
            r'^([A-Z][a-z]+)\s*:',  # Name: at start of line
            r'^([A-Z][a-z]+)\s+\d+:',  # Name followed by timestamp
            r'\[([A-Z][a-z]+)\]',  # [Name] pattern
        ]
        
        participants = set()
        for pattern in participant_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            participants.update(matches)
        
        return list(participants)
    
    def _extract_chat_messages(self, text: str) -> list:
        """Extract chat messages"""
        import re
        
        # Split by common chat separators
        messages = re.split(r'\n\d+:\d+', text)
        
        # Filter out empty messages and very short ones
        messages = [msg.strip() for msg in messages if len(msg.strip()) > 3]
        
        return messages[:20]  # Limit to first 20 messages
    
    def _extract_chat_timestamps(self, text: str) -> list:
        """Extract chat timestamps"""
        import re
        
        timestamp_patterns = [
            r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',
            r'\d{1,2}\.\d{2}\s*(?:AM|PM|am|pm)?',
            r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}'
        ]
        
        timestamps = []
        for pattern in timestamp_patterns:
            matches = re.findall(pattern, text)
            timestamps.extend(matches)
        
        return list(set(timestamps))[:10]  # Limit to first 10 unique timestamps

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
