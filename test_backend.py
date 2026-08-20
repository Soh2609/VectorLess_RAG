import sys
import os
from dotenv import load_dotenv

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pipeline():
    print("Step 1: Loading environment variables...")
    load_dotenv()
    
    try:
        from vectorless_rag.config import settings
        print("✅ Config loaded successfully.")
        print(f"   Gemini Model: {settings.gemini_model}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

    print("\nStep 2: Checking PageIndex connectivity...")
    try:
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=settings.pageindex_api_key)
        print("✅ PageIndex client initialized.")
    except Exception as e:
        print(f"❌ PageIndex connection failed: {e}")
        return False

    print("\nStep 3: Checking Gemini API connectivity...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content("Respond with exactly the word 'OK'.")
        print(f"✅ Gemini connection successful. Response: '{response.text.strip()}'")
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

    print("\n🎉 All basic connectivity tests passed! You are ready to run the Streamlit app.")
    return True

if __name__ == "__main__":
    test_pipeline()
