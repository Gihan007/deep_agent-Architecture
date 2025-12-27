#!/usr/bin/env python3
"""
Test OpenAI API key
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

print("🔍 Testing OpenAI API key...")

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"Key loaded: {openai_key[:25]}...")
    try:
        llm = ChatOpenAI(model="gpt-4", api_key=openai_key)
        response = llm.invoke("Say 'OpenAI API working perfectly' in exactly those words")
        print(f"✅ OpenAI API working: {response.content}")
    except Exception as e:
        print(f"❌ OpenAI API failed: {str(e)}")
else:
    print("❌ No OPENAI_API_KEY found")