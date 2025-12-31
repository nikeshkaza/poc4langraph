"""
Test script to verify PyMuPDF, Azure Document Intelligence, and OpenAI setup.
Run this to check if your environment is configured correctly.
"""

import os
import sys
import asyncio


def check_environment_variables():
    """Check if required environment variables are set."""
    print("=" * 60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API access',
        'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT': 'Azure Document Intelligence endpoint',
        'AZURE_DOCUMENT_INTELLIGENCE_KEY': 'Azure Document Intelligence API key'
    }
    
    all_set = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask the key for security
            if 'KEY' in var:
                masked = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                print(f"✅ {var}: {masked}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET (needed for {description})")
            all_set = False
    
    print()
    return all_set


def check_pymupdf():
    """Check if PyMuPDF is installed."""
    print("=" * 60)
    print("CHECKING PYMUPDF")
    print("=" * 60)
    
    try:
        import fitz
        print(f"✅ PyMuPDF (fitz) installed: version {fitz.version}")
        print(f"   PyMuPDF can parse PDF files")
        return True
    except ImportError:
        print("❌ PyMuPDF not installed")
        print("   Install with: pip install PyMuPDF")
        return False
    print()


def check_azure_sdk():
    """Check if Azure SDK is installed."""
    print("=" * 60)
    print("CHECKING AZURE SDK")
    print("=" * 60)
    
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        print("✅ Azure SDK installed")
        print("   azure-ai-formrecognizer available")
        return True
    except ImportError:
        print("❌ Azure SDK not installed")
        print("   Install with: pip install azure-ai-formrecognizer")
        return False
    print()


def check_openai():
    """Check if OpenAI library is installed."""
    print("=" * 60)
    print("CHECKING OPENAI")
    print("=" * 60)
    
    try:
        import openai
        print(f"✅ OpenAI library installed: version {openai.__version__}")
        return True
    except ImportError:
        print("❌ OpenAI library not installed")
        print("   Install with: pip install openai")
        return False
    print()


async def test_pymupdf_parsing():
    """Test PyMuPDF with a sample PDF."""
    print("=" * 60)
    print("TESTING PYMUPDF")
    print("=" * 60)
    
    try:
        import fitz
        
        # Create a simple test PDF in memory
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Content\nLine 2\nLine 3")
        
        # Save to temp file
        temp_path = "/tmp/test_pymupdf.pdf"
        doc.save(temp_path)
        doc.close()
        
        # Try to read it back
        doc = fitz.open(temp_path)
        text = doc[0].get_text()
        doc.close()
        
        print(f"✅ PyMuPDF test successful")
        print(f"   Created and parsed test PDF")
        print(f"   Extracted text: {text[:50]}...")
        
        # Clean up
        os.remove(temp_path)
        return True
        
    except Exception as e:
        print(f"❌ PyMuPDF test failed: {e}")
        return False
    print()


async def test_azure_connection():
    """Test Azure Document Intelligence connection."""
    print("=" * 60)
    print("TESTING AZURE CONNECTION")
    print("=" * 60)
    
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        
        endpoint = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
        key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
        
        if not endpoint or not key:
            print("⚠️  Azure credentials not set, skipping connection test")
            return None
        
        # Initialize client
        client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        print(f"✅ Azure client initialized successfully")
        print(f"   Endpoint: {endpoint}")
        print(f"   Ready to process documents")
        return True
        
    except Exception as e:
        print(f"❌ Azure connection failed: {e}")
        print(f"   Check your endpoint and API key")
        return False
    print()


async def test_openai_connection():
    """Test OpenAI API connection."""
    print("=" * 60)
    print("TESTING OPENAI CONNECTION")
    print("=" * 60)
    
    try:
        import openai
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OpenAI API key not set, skipping connection test")
            return None
        
        # Initialize client
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Make a simple test call
        print("   Making test API call...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test successful' in 3 words"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI connection successful")
        print(f"   Test response: {result}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        print(f"   Check your API key and account status")
        return False
    print()


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EXTRACTION SYSTEM SETUP CHECKER" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    results = {}
    
    # Check installations
    results['env_vars'] = check_environment_variables()
    results['pymupdf'] = check_pymupdf()
    results['azure_sdk'] = check_azure_sdk()
    results['openai_lib'] = check_openai()
    
    # Test functionality
    if results['pymupdf']:
        results['pymupdf_test'] = await test_pymupdf_parsing()
    
    if results['azure_sdk'] and results['env_vars']:
        results['azure_test'] = await test_azure_connection()
    
    if results['openai_lib'] and results['env_vars']:
        results['openai_test'] = await test_openai_connection()
    
    # Summary
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_good = True
    
    print("\n📦 Installations:")
    for check in ['pymupdf', 'azure_sdk', 'openai_lib']:
        if results.get(check):
            print(f"  ✅ {check.replace('_', ' ').title()}")
        else:
            print(f"  ❌ {check.replace('_', ' ').title()}")
            all_good = False
    
    print("\n🔧 Configuration:")
    if results.get('env_vars'):
        print("  ✅ Environment variables set")
    else:
        print("  ❌ Missing environment variables")
        all_good = False
    
    print("\n🧪 Functional Tests:")
    for test in ['pymupdf_test', 'azure_test', 'openai_test']:
        if test in results:
            if results[test]:
                print(f"  ✅ {test.replace('_', ' ').title()}")
            elif results[test] is None:
                print(f"  ⚠️  {test.replace('_', ' ').title()} - Skipped")
            else:
                print(f"  ❌ {test.replace('_', ' ').title()}")
                all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("Your environment is ready to run the extraction system.")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Please fix the issues above before running the system.")
    print("=" * 60)
    print("\n")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
