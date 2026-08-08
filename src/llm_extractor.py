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
        # Detect document type first
        doc_type = self._detect_document_type(text)
        
        # Get document type-specific prompt
        type_specific_instructions = self._get_type_specific_instructions(doc_type)
        
        # Get few-shot examples
        few_shot_examples = self._get_few_shot_examples(doc_type)
        
        prompt = f"""
You are an expert document extraction system specializing in extracting structured information from various document types with high accuracy.

{type_specific_instructions}

{few_shot_examples}

Document Text to Extract From:
{text}

{context}

Extract the following fields with high precision:
- document_type: Type of document (insurance_claim, invoice, receipt, contract, form, chat_conversation, letter, report, certificate, statement, or unknown)
- date: All dates mentioned in the document (return as array if multiple)
- amount: All monetary amounts mentioned (return as array if multiple)
- names: All person and company names mentioned (return as array)
- addresses: All addresses mentioned (return as array)
- phone_numbers: All phone numbers mentioned (return as array)
- email_addresses: All email addresses mentioned (return as array)
- reference_numbers: All reference numbers, invoice numbers, order IDs, or other identifiers (return as array)
- total_amount: The total/sum amount if explicitly stated
- items: For invoices/receipts, extract as array of objects with: name, quantity, unit_price, total_price
- other_fields: Any other important key-value pairs or information not covered above (as object)

IMPORTANT EXTRACTION RULES:
1. Be precise and extract only what is explicitly stated in the document
2. For dates, preserve the original format but standardize to YYYY-MM-DD when possible
3. For amounts, include currency symbols and preserve decimal precision
4. For names, distinguish between person names and company names
5. For phone numbers, include country codes if present
6. For items, extract all line items with their quantities and prices
7. If a field is not found or cannot be confidently extracted, set it to null
8. Return arrays for fields that can have multiple values (dates, amounts, names, etc.)

Return the result as a JSON object with the extracted fields.
"""
        return prompt
    
    def _get_type_specific_instructions(self, doc_type: str) -> str:
        """Get document type-specific extraction instructions"""
        instructions = {
            'invoice': '''
INVOICE EXTRACTION INSTRUCTIONS:
- Focus on extracting invoice number, invoice date, due date, vendor information, billing address, shipping address
- Extract all line items with: item name/description, quantity, unit price, line total
- Identify subtotal, tax amounts, discounts, and final total
- Look for payment terms, payment method, and bank details
- Extract any purchase order numbers referenced
''',
            'receipt': '''
RECEIPT EXTRACTION INSTRUCTIONS:
- Focus on extracting receipt number, transaction date, time, merchant name, location
- Extract all purchased items with: item name, quantity, unit price, line total
- Identify subtotal, tax amounts, tip/gratuity, and final total
- Look for payment method (cash, card, etc.) and last 4 digits of card
- Extract any loyalty program information or customer ID
''',
            'contract': '''
CONTRACT EXTRACTION INSTRUCTIONS:
- Focus on extracting contract title, contract number, effective date, expiration date
- Extract all parties involved (names, roles, addresses)
- Identify key terms: contract value, payment terms, deliverables, milestones
- Look for signatures, signatories, and signing dates
- Extract any amendment numbers or references
''',
            'form': '''
FORM EXTRACTION INSTRUCTIONS:
- Focus on extracting form title, form number, submission date
- Extract all form fields with their labels and values
- Identify submitter information (name, contact details, signature)
- Look for any reference numbers, case numbers, or application IDs
- Extract checkboxes or selection indicators
''',
            'chat_conversation': '''
CHAT CONVERSATION EXTRACTION INSTRUCTIONS:
- Focus on extracting all participants in the conversation
- Extract messages with timestamps and senders
- Identify key information shared: dates, amounts, names, locations
- Look for any decisions, agreements, or action items
- Extract any contact information or references mentioned
''',
            'letter': '''
LETTER EXTRACTION INSTRUCTIONS:
- Focus on extracting sender and recipient information (names, addresses)
- Extract letter date and reference numbers
- Identify subject line and key topics discussed
- Look for any deadlines, dates, or action items mentioned
- Extract signatures and signatories
''',
            'report': '''
REPORT EXTRACTION INSTRUCTIONS:
- Focus on extracting report title, report date, author/organization
- Extract key metrics, figures, and statistics
- Identify time periods covered in the report
- Look for any conclusions, recommendations, or action items
- Extract any reference numbers or report IDs
''',
            'certificate': '''
CERTIFICATE EXTRACTION INSTRUCTIONS:
- Focus on extracting certificate title, certificate number, issue date
- Extract recipient name and issuing organization
- Identify expiration date if applicable
- Look for any grades, scores, or achievement levels
- Extract signatures and authority information
''',
            'statement': '''
STATEMENT EXTRACTION INSTRUCTIONS:
- Focus on extracting statement period (start date, end date)
- Extract account numbers and customer information
- Identify opening balance, transactions, and closing balance
- Look for payment due dates and minimum payment amounts
- Extract any reference numbers or transaction IDs
''',
            'insurance_claim': '''
INSURANCE CLAIM EXTRACTION INSTRUCTIONS:
- Focus on extracting claim-specific fields: claim number, risk reference, transaction reference
- Extract insured information: company name, address, contact details
- Extract reinsured information: company name, address
- Extract contract details: contract period, type, limits, date of loss, policy period
- Extract loss information: loss name, location, claimant name, incident description
- Extract financial data: incurred loss, paid loss, retention, expenses (with currency)
- Extract all key-value pairs present in the document
- Preserve the exact structure of financial figures
- Extract contact information: phone numbers, email addresses
''',
            'unknown': '''
GENERAL DOCUMENT EXTRACTION INSTRUCTIONS:
- Extract all identifiable information regardless of document type
- Focus on names, dates, amounts, addresses, and contact information
- Look for any reference numbers or identifiers
- Extract any key-value pairs or structured data
- Identify the document purpose and main topics
'''
        }
        return instructions.get(doc_type, instructions['unknown'])
    
    def _get_few_shot_examples(self, doc_type: str) -> str:
        """Get few-shot examples for the document type"""
        examples = {
            'invoice': '''
EXAMPLE EXTRACTION - INVOICE:
Input: "INVOICE #INV-2024-001
Date: 2024-01-15
Due Date: 2024-02-15
From: TechCorp Inc.
123 Business Ave, Suite 100
San Francisco, CA 94105
To: ABC Company
456 Industry Blvd
New York, NY 10001

Item 1: Software License - 5 units @ $100.00 = $500.00
Item 2: Support Services - 10 hours @ $50.00 = $500.00

Subtotal: $1,000.00
Tax (8%): $80.00
Total: $1,080.00

Payment Terms: Net 30
Bank: Chase Bank, Account: ****1234"

Output: {
  "document_type": "invoice",
  "date": ["2024-01-15", "2024-02-15"],
  "amount": ["$500.00", "$500.00", "$1,000.00", "$80.00", "$1,080.00"],
  "names": ["TechCorp Inc.", "ABC Company"],
  "addresses": ["123 Business Ave, Suite 100, San Francisco, CA 94105", "456 Industry Blvd, New York, NY 10001"],
  "phone_numbers": null,
  "email_addresses": null,
  "reference_numbers": ["INV-2024-001"],
  "total_amount": "$1,080.00",
  "items": [
    {"name": "Software License", "quantity": 5, "unit_price": "$100.00", "total_price": "$500.00"},
    {"name": "Support Services", "quantity": 10, "unit_price": "$50.00", "total_price": "$500.00"}
  ],
  "other_fields": {
    "payment_terms": "Net 30",
    "tax_rate": "8%",
    "bank": "Chase Bank",
    "account": "****1234"
  }
}
''',
            'receipt': '''
EXAMPLE EXTRACTION - RECEIPT:
Input: "RECEIPT #R-4567
Date: 01/20/2024
Time: 14:35
Merchant: Coffee Shop
123 Main Street
Anytown, USA

2x Latte - $4.50 each = $9.00
1x Muffin - $3.50 = $3.50

Subtotal: $12.50
Tax: $1.00
Tip: $2.50
Total: $16.00

Paid: Visa ****4321

Thank you for your visit!
Loyalty Member: LM12345"

Output: {
  "document_type": "receipt",
  "date": ["01/20/2024"],
  "amount": ["$9.00", "$3.50", "$12.50", "$1.00", "$2.50", "$16.00"],
  "names": ["Coffee Shop"],
  "addresses": ["123 Main Street, Anytown, USA"],
  "phone_numbers": null,
  "email_addresses": null,
  "reference_numbers": ["R-4567"],
  "total_amount": "$16.00",
  "items": [
    {"name": "Latte", "quantity": 2, "unit_price": "$4.50", "total_price": "$9.00"},
    {"name": "Muffin", "quantity": 1, "unit_price": "$3.50", "total_price": "$3.50"}
  ],
  "other_fields": {
    "time": "14:35",
    "payment_method": "Visa",
    "card_last_4": "4321",
    "loyalty_member": "LM12345"
  }
}
''',
            'contract': '''
EXAMPLE EXTRACTION - CONTRACT:
Input: "SERVICE AGREEMENT
Contract #: SA-2024-789
Effective: January 1, 2024
Expiration: December 31, 2024

Between:
Provider: XYZ Services LLC
100 Provider Lane
Chicago, IL 60601

And:
Client: Global Corp
200 Client Road
New York, NY 10002

Services: Software Development
Contract Value: $50,000
Payment Terms: Monthly installments of $5,000
Deliverables: 10 software modules

Signed: John Smith (Provider)
Signed: Jane Doe (Client)
Date: December 15, 2023"

Output: {
  "document_type": "contract",
  "date": ["January 1, 2024", "December 31, 2024", "December 15, 2023"],
  "amount": ["$50,000", "$5,000"],
  "names": ["XYZ Services LLC", "Global Corp", "John Smith", "Jane Doe"],
  "addresses": ["100 Provider Lane, Chicago, IL 60601", "200 Client Road, New York, NY 10002"],
  "phone_numbers": null,
  "email_addresses": null,
  "reference_numbers": ["SA-2024-789"],
  "total_amount": "$50,000",
  "items": null,
  "other_fields": {
    "services": "Software Development",
    "payment_terms": "Monthly installments of $5,000",
    "deliverables": "10 software modules",
    "signatories": ["John Smith", "Jane Doe"]
  }
}
''',
            'chat_conversation': '''
EXAMPLE EXTRACTION - CHAT CONVERSATION:
Input: "10:30 AM - Alice: Hey, can we meet tomorrow?
10:32 AM - Bob: Sure, what time works for you?
10:33 AM - Alice: How about 2 PM at the coffee shop on Main St?
10:35 AM - Bob: Perfect! I'll bring the documents.
10:36 AM - Alice: Great, see you there. My number is 555-1234 if you need to reach me."

Output: {
  "document_type": "chat_conversation",
  "date": null,
  "amount": null,
  "names": ["Alice", "Bob"],
  "addresses": ["coffee shop on Main St"],
  "phone_numbers": ["555-1234"],
  "email_addresses": null,
  "reference_numbers": null,
  "total_amount": null,
  "items": null,
  "other_fields": {
    "participants": ["Alice", "Bob"],
    "meeting_time": "2 PM",
    "meeting_location": "coffee shop on Main St",
    "action_items": ["Bob to bring documents"]
  }
}
''',
            'insurance_claim': '''
EXAMPLE EXTRACTION - INSURANCE CLAIM:
Input: "CLAIM ADVICE
Risk Reference: C2ZX000598002
Claim Number: CA109114/2/7
Transaction Ref.: 38631800
Date: 15 November 2022

Insured: Terra Nova Insurance Company Limited
Address: c/o Markel International Ins Co Ltd, The Markel Building, 49 Leadenhall Street, London EC3A 2

Reinsured: Northbridge General Insurance Corporation

Contract Period: 01/01/1996 to 12/31/1996
Type: General Liability
Limits: 3,000,000 XS 4,000,000
Date of Loss: 11/Sep/1999
Loss Name: Gordon Whitehead
Location: Ontario
Claimant Name: Adam, Jack

Incurred Loss: 3,144,655.87
Paid Loss: 3,677,715.87
Retention: 6,000,000.00
Incurred Expenses: 75,675.04

Contact: Rhonda Bass, Phone: 1 206 621 2431, Email: Rhonda.S.Bass@guycarp.com"

Output: {
  "document_type": "insurance_claim",
  "date": ["15 November 2022", "01/01/1996", "12/31/1996", "11/Sep/1999"],
  "amount": ["3,144,655.87", "3,677,715.87", "6,000,000.00", "75,675.04"],
  "names": ["Terra Nova Insurance Company Limited", "Markel International Ins Co Ltd", "Northbridge General Insurance Corporation", "Gordon Whitehead", "Adam, Jack", "Rhonda Bass"],
  "addresses": ["The Markel Building, 49 Leadenhall Street, London EC3A 2"],
  "phone_numbers": ["1 206 621 2431"],
  "email_addresses": ["Rhonda.S.Bass@guycarp.com"],
  "reference_numbers": ["C2ZX000598002", "CA109114/2/7", "38631800"],
  "total_amount": "3,144,655.87",
  "items": null,
  "other_fields": {
    "risk_reference": "C2ZX000598002",
    "claim_number": "CA109114/2/7",
    "transaction_ref": "38631800",
    "insured": "Terra Nova Insurance Company Limited",
    "reinsured": "Northbridge General Insurance Corporation",
    "contract_period": "01/01/1996 to 12/31/1996",
    "type": "General Liability",
    "limits": "3,000,000 XS 4,000,000",
    "date_of_loss": "11/Sep/1999",
    "loss_name": "Gordon Whitehead",
    "location": "Ontario",
    "claimant_name": "Adam, Jack",
    "incurred_loss": "3,144,655.87",
    "paid_loss": "3,677,715.87",
    "retention": "6,000,000.00",
    "incurred_expenses": "75,675.04",
    "contact": "Rhonda Bass"
  }
}
'''
        }
        return examples.get(doc_type, '')
    
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
        
        type_keywords = {
            'insurance_claim': ['claim', 'insurance', 'reinsured', 'incurred loss', 'paid loss', 'retention', 'claim advice', 'loss due', 'policy period', 'date of loss'],
            'invoice': ['invoice', 'bill', 'rechnung', 'fatura'],
            'receipt': ['receipt', 'quittung', 'fiş', 'receipt'],
            'contract': ['contract', 'agreement', 'vertrag', 'sözleşme'],
            'form': ['form', 'application', 'antrag', 'başvuru'],
            'letter': ['letter', 'brief', 'mektup'],
            'report': ['report', 'bericht', 'rapor'],
            'certificate': ['certificate', 'zertifikat', 'sertifika'],
            'statement': ['statement', 'kontoauszug', 'ekstre']
        }
        
        # Check for insurance claim first (highest priority for structured documents)
        for doc_type, keywords in type_keywords.items():
            if doc_type == 'insurance_claim':
                if any(keyword in text_lower for keyword in keywords):
                    return doc_type
        
        # Check for chat/conversation last (only if no other type matches)
        if self._is_chat_conversation(text):
            return 'chat_conversation'
        
        # Check other document types
        for doc_type, keywords in type_keywords.items():
            if doc_type != 'insurance_claim':
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
            # Enhanced key-value pair extraction for structured documents
            # More precise pattern to avoid noise
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Pattern for "Key: Value" or "Key : Value"
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Filter out noise - key should be meaningful
                        if len(key) >= 2 and len(value) >= 1:
                            # Skip if key is too generic
                            skip_keys = ['the', 'and', 'or', 'to', 'for', 'with', 'by', 'of', 'in', 'on', 'at', 'from', 'this', 'that', 'a', 'an']
                            if key.lower() not in skip_keys:
                                # Clean up value
                                value = re.sub(r'[.,;:]+$', '', value)
                                # Only add if value is not just whitespace or numbers
                                if value and not value.isspace():
                                    other_fields[key] = value
        
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
