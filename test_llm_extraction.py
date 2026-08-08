"""
Test script for LLM extraction with sample documents
"""
import sys
import os
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_extractor import LLMExtractor

def test_invoice_extraction():
    """Test invoice extraction"""
    print("\n" + "="*80)
    print("TEST 1: INVOICE EXTRACTION")
    print("="*80)
    
    invoice_text = """
    INVOICE #INV-2024-001
    Date: 2024-01-15
    Due Date: 2024-02-15
    From: TechCorp Inc.
    123 Business Ave, Suite 100
    San Francisco, CA 94105
    Email: billing@techcorp.com
    Phone: +1-555-123-4567
    
    To: ABC Company
    456 Industry Blvd
    New York, NY 10001
    
    Item 1: Software License - 5 units @ $100.00 = $500.00
    Item 2: Support Services - 10 hours @ $50.00 = $500.00
    
    Subtotal: $1,000.00
    Tax (8%): $80.00
    Total: $1,080.00
    
    Payment Terms: Net 30
    Bank: Chase Bank, Account: ****1234
    """
    
    extractor = LLMExtractor()
    results = extractor.extract(invoice_text)
    
    print("\nInput Document:")
    print(invoice_text)
    
    print("\nExtraction Results:")
    print(json.dumps(results.get('llm_results', {}), indent=2))
    
    return results

def test_receipt_extraction():
    """Test receipt extraction"""
    print("\n" + "="*80)
    print("TEST 2: RECEIPT EXTRACTION")
    print("="*80)
    
    receipt_text = """
    RECEIPT #R-4567
    Date: 01/20/2024
    Time: 14:35
    Merchant: Coffee Shop
    123 Main Street
    Anytown, USA
    Phone: (555) 987-6543
    
    2x Latte - $4.50 each = $9.00
    1x Muffin - $3.50 = $3.50
    
    Subtotal: $12.50
    Tax: $1.00
    Tip: $2.50
    Total: $16.00
    
    Paid: Visa ****4321
    
    Thank you for your visit!
    Loyalty Member: LM12345
    """
    
    extractor = LLMExtractor()
    results = extractor.extract(receipt_text)
    
    print("\nInput Document:")
    print(receipt_text)
    
    print("\nExtraction Results:")
    print(json.dumps(results.get('llm_results', {}), indent=2))
    
    return results

def test_chat_extraction():
    """Test chat conversation extraction"""
    print("\n" + "="*80)
    print("TEST 3: CHAT CONVERSATION EXTRACTION")
    print("="*80)
    
    chat_text = """
    10:30 AM - Alice: Hey, can we meet tomorrow?
    10:32 AM - Bob: Sure, what time works for you?
    10:33 AM - Alice: How about 2 PM at the coffee shop on Main St?
    10:35 AM - Bob: Perfect! I'll bring the documents.
    10:36 AM - Alice: Great, see you there. My number is 555-1234 if you need to reach me.
    """
    
    extractor = LLMExtractor()
    results = extractor.extract(chat_text)
    
    print("\nInput Document:")
    print(chat_text)
    
    print("\nExtraction Results:")
    print(json.dumps(results.get('llm_results', {}), indent=2))
    
    return results

def test_contract_extraction():
    """Test contract extraction"""
    print("\n" + "="*80)
    print("TEST 4: CONTRACT EXTRACTION")
    print("="*80)
    
    contract_text = """
    SERVICE AGREEMENT
    Contract #: SA-2024-789
    Effective: January 1, 2024
    Expiration: December 31, 2024
    
    Between:
    Provider: XYZ Services LLC
    100 Provider Lane
    Chicago, IL 60601
    Email: info@xyzservices.com
    
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
    Date: December 15, 2023
    """
    
    extractor = LLMExtractor()
    results = extractor.extract(contract_text)
    
    print("\nInput Document:")
    print(contract_text)
    
    print("\nExtraction Results:")
    print(json.dumps(results.get('llm_results', {}), indent=2))
    
    return results

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("LLM EXTRACTION TEST SUITE")
    print("="*80)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("Tests will use fallback rule-based extraction")
        print("For full LLM testing, set OPENAI_API_KEY in .env file\n")
    
    try:
        # Run tests
        test_invoice_extraction()
        test_receipt_extraction()
        test_chat_extraction()
        test_contract_extraction()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
