#!/usr/bin/env python3
"""
Test script for Deep Agent Research Assistant
Run basic validation tests for all components
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import DeepAgentResearchAssistant


async def test_basic_functionality():
    """Test basic agent functionality"""
    print("🧪 Testing Deep Agent Research Assistant...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Check environment
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"❌ Missing required environment variables: {missing_keys}")
        print("Please set them in your .env file and run: source .env")
        return False

    print("✅ Environment variables configured")

    # Initialize assistant
    try:
        assistant = DeepAgentResearchAssistant()
        print("✅ Assistant initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        return False

    # Test basic research (short topic for testing)
    test_topic = "What is Python?"
    print(f"🔍 Testing research on: {test_topic}")

    try:
        result = await assistant.research_topic(test_topic, "basic")
        if result["success"]:
            print("✅ Research completed successfully")
            print(f"Response length: {len(result['response'])} characters")
            # Show first 200 characters
            preview = result['response'][:200] + "..." if len(result['response']) > 200 else result['response']
            print(f"Preview: {preview}")
        else:
            print(f"❌ Research failed: {result['error']}")
            # If it's an API error, suggest checking keys
            if "401" in str(result['error']) or "authentication" in str(result['error']).lower():
                print("💡 This might be an API key issue. Please verify your API keys are correct and active.")
            return False
    except Exception as e:
        print(f"❌ Research test failed: {e}")
        return False

    # Test workspace files
    files = assistant.list_workspace_files()
    if files:
        print(f"✅ Generated {len(files)} workspace files: {files[:3]}...")
    else:
        print("ℹ️  No workspace files generated (expected for basic test)")

    # Test memory tools
    try:
        # Store memory
        store_result = assistant.store_memory_tool("test_key", "test_content")
        print(f"✅ Memory storage: {store_result}")

        # Retrieve memory
        retrieve_result = assistant.retrieve_memory_tool("test_key")
        if "test_content" in str(retrieve_result):
            print("✅ Memory retrieval successful")
        else:
            print(f"❌ Memory retrieval failed: {retrieve_result}")
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

    print("\n🎉 All tests passed! Deep Agent is ready to use.")
    return True


def test_imports():
    """Test that all required modules can be imported"""
    print("📦 Testing imports...")

    try:
        import deepagents
        print("✅ deepagents imported")
    except ImportError:
        print("❌ deepagents not installed. Run: pip install deepagents")
        return False

    try:
        from tavily import TavilyClient
        print("✅ tavily-python imported")
    except ImportError:
        print("❌ tavily-python not installed. Run: pip install tavily-python")
        return False

    try:
        import langchain
        print("✅ langchain imported")
    except ImportError:
        print("❌ langchain not installed. Run: pip install langchain")
        return False

    try:
        import langgraph
        print("✅ langgraph imported")
    except ImportError:
        print("❌ langgraph not installed. Run: pip install langgraph")
        return False

    return True


if __name__ == "__main__":
    print("🚀 Deep Agent Research Assistant - Test Suite\n")

    # Test imports first
    if not test_imports():
        sys.exit(1)

    print()

    # Run async tests
    try:
        success = asyncio.run(test_basic_functionality())
        if success:
            print("\n💡 Next steps:")
            print("  1. Run: python -m src.main 'your research topic'")
            print("  2. Try different depths: --depth basic|intermediate|advanced")
            print("  3. Explore workspace files: python -m src.main --list-files 'dummy'")
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)