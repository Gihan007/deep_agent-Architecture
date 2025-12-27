#!/usr/bin/env python3
"""
Debug agent authentication issue
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

# Load environment variables
load_dotenv()

print("🔍 Debugging agent authentication...")

# Set environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(f"OpenAI key set: {os.environ.get('OPENAI_API_KEY')[:25] if os.environ.get('OPENAI_API_KEY') else 'None'}...")

@tool
def simple_search_tool(query: str) -> str:
    """Simple search tool for testing"""
    return f"Mock search result for: {query}"

print("Creating LLM...")
llm = ChatOpenAI(model="gpt-4")
print("✅ LLM created")

print("Creating InMemoryStore...")
store = InMemoryStore()
print("✅ Store created")

print("Creating React Agent...")
try:
    agent = create_react_agent(
        llm,
        tools=[simple_search_tool],
        prompt="You are a helpful assistant."
    )
    print("✅ React Agent created successfully!")

    print("Testing agent invocation...")
    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    print(f"✅ Agent response: {result['messages'][-1].content}")

except Exception as e:
    print(f"❌ Agent failed: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()