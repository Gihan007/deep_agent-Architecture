#!/usr/bin/env python3
"""
Test Tavily API key specifically
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

print("🔍 Testing Tavily API key...")

tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    print(f"Key loaded: {tavily_key[:25]}...")
    try:
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query="test quantum computing", max_results=1)
        print("✅ Tavily API working!")
        print(f"Found {len(results.get('results', []))} results")
    except Exception as e:
        print(f"❌ Tavily API failed: {str(e)}")
        if "invalid" in str(e).lower():
            print("💡 The API key appears to be invalid or expired")
        elif "quota" in str(e).lower():
            print("💡 You may have exceeded your API quota")
else:
    print("❌ No TAVILY_API_KEY found")