#!/usr/bin/env python3
"""Test script to verify OpenRouter API key and check available models."""

import os
import sys
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is loaded."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("   Please create a .env file with: OPENROUTER_API_KEY=your_key_here")
        return None
    else:
        print(f"✅ API Key found (length: {len(api_key)})")
        return api_key

def list_available_models(api_key):
    """List available models on OpenRouter."""
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Try to get models list
        import requests
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        if response.status_code == 200:
            models = response.json().get("data", [])
            print(f"\n✅ Found {len(models)} available models")
            
            # Look for GPT models
            gpt_models = [m for m in models if "gpt" in m.get("id", "").lower()]
            print(f"\n📋 GPT Models (showing first 5):")
            for model in gpt_models[:5]:
                print(f"   - {model.get('id')}")
            
            # Look for Claude models
            claude_models = [m for m in models if "claude" in m.get("id", "").lower()]
            print(f"\n📋 Claude Models (showing first 5):")
            for model in claude_models[:5]:
                print(f"   - {model.get('id')}")
            
            # Look for Gemini models
            gemini_models = [m for m in models if "gemini" in m.get("id", "").lower()]
            print(f"\n📋 Gemini Models (showing first 5):")
            for model in gemini_models[:5]:
                print(f"   - {model.get('id')}")
            
            return models
        else:
            print(f"❌ ERROR: Failed to fetch models (status: {response.status_code})")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        return None

def test_model_name(api_key, model_id):
    """Test if a specific model name works."""
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Say 'test' and nothing else."}
            ],
            max_tokens=5,
        )
        
        result = response.choices[0].message.content.strip()
        print(f"   ✅ Model '{model_id}' works! Response: '{result}'")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            print(f"   ❌ Model '{model_id}' not found or unavailable")
        else:
            print(f"   ⚠️  Model '{model_id}' error: {error_msg[:100]}")
        return False

def main():
    print("=" * 60)
    print("OpenRouter API Key and Model Verification")
    print("=" * 60)
    
    # Test API key
    api_key = test_api_key()
    if not api_key:
        sys.exit(1)
    
    # List available models
    print("\n" + "=" * 60)
    print("Fetching available models from OpenRouter...")
    print("=" * 60)
    
    try:
        import requests
    except ImportError:
        print("⚠️  'requests' library not found. Installing...")
        os.system("pip install requests -q")
        import requests
    
    models = list_available_models(api_key)
    
    # Test current model names
    print("\n" + "=" * 60)
    print("Testing Current Model Names")
    print("=" * 60)
    
    current_models = {
        "GPT": "openai/gpt-5",
        "Claude": "anthropic/claude-sonnet-4.5",
        "Gemini": "google/gemini-2.5-pro"
    }
    
    for name, model_id in current_models.items():
        print(f"\nTesting {name} ({model_id}):")
        test_model_name(api_key, model_id)
    
    # Suggest alternatives
    print("\n" + "=" * 60)
    print("Suggested Model Names (if current ones don't work)")
    print("=" * 60)
    print("Common alternatives:")
    print("  GPT: openai/gpt-4o, openai/gpt-4-turbo")
    print("  Claude: anthropic/claude-3.5-sonnet, anthropic/claude-3-opus")
    print("  Gemini: google/gemini-pro-1.5, google/gemini-1.5-pro")

if __name__ == "__main__":
    main()

