#!/usr/bin/env python3
"""
Quick test of OpenAI and Tavily APIs
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# Load environment variables
load_dotenv()

print("🔍 Testing OpenAI and Tavily APIs...")

# Test OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    try:
        llm = ChatOpenAI(model="gpt-4", api_key=openai_key)
        response = llm.invoke("Say 'OpenAI working' in two words")
        print(f"✅ OpenAI: {response.content}")
    except Exception as e:
        print(f"❌ OpenAI failed: {str(e)[:100]}")
else:
    print("❌ No OpenAI key")

# Test Tavily
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    try:
        client = TavilyClient(api_key=tavily_key)
        results = client.search(query="test", max_results=1)
        print("✅ Tavily: Working")
    except Exception as e:
        print(f"❌ Tavily failed: {str(e)[:100]}")
else:
    print("❌ No Tavily key")

print("🎯 Ready to use OpenAI + Tavily!")