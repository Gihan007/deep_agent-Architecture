#!/usr/bin/env python3
"""
Example usage scripts for Deep Agent Research Assistant
Demonstrates different research scenarios and agent capabilities
"""

import asyncio
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import DeepAgentResearchAssistant


async def example_basic_research():
    """Example: Basic research on a simple topic"""
    print("📖 Example 1: Basic Research")
    print("=" * 40)

    assistant = DeepAgentResearchAssistant()

    result = await assistant.research_topic(
        "What are the main benefits of using Python for data science?",
        "basic"
    )

    print(result["response"])
    print("\n" + "=" * 40 + "\n")


async def example_intermediate_research():
    """Example: Intermediate research with planning and subagents"""
    print("🔍 Example 2: Intermediate Research with Planning")
    print("=" * 50)

    assistant = DeepAgentResearchAssistant()

    result = await assistant.research_topic(
        "Compare and contrast REST APIs vs GraphQL vs gRPC for modern web services",
        "intermediate"
    )

    print(result["response"])
    print("\n" + "=" * 50 + "\n")


async def example_advanced_research():
    """Example: Advanced research with deep analysis and memory"""
    print("🧠 Example 3: Advanced Research with Memory")
    print("=" * 45)

    assistant = DeepAgentResearchAssistant()

    # First research - store in memory
    print("Phase 1: Research microservices architecture")
    result1 = await assistant.research_topic(
        "What is microservices architecture and what are its key principles?",
        "intermediate"
    )
    print("✓ Completed and stored in memory")

    # Second research - build on previous knowledge
    print("\nPhase 2: Research serverless in context of microservices")
    result2 = await assistant.research_topic(
        "How does serverless computing complement or conflict with microservices architecture?",
        "advanced"
    )

    print(result2["response"])
    print("\n" + "=" * 45 + "\n")


async def example_codebase_analysis():
    """Example: Codebase analysis using custom tools"""
    print("💻 Example 4: Codebase Analysis")
    print("=" * 35)

    assistant = DeepAgentResearchAssistant()

    # This would normally analyze a real repository
    # For demo, we'll use a mock analysis
    analysis_result = assistant.analyze_codebase_tool(
        "https://github.com/langchain-ai/deepagents",
        ["architecture", "dependencies", "testing"]
    )

    print("Codebase Analysis Result:")
    print(f"Repository: {analysis_result['repo_url']}")
    print(f"Architecture: {analysis_result['analysis']['architecture']}")
    print(f"Key Dependencies: {', '.join(analysis_result['analysis']['dependencies'])}")
    print(f"Recommendations: {', '.join(analysis_result['analysis']['recommendations'])}")

    print("\n" + "=" * 35 + "\n")


async def example_memory_operations():
    """Example: Direct memory operations"""
    print("🗃️  Example 5: Memory Operations")
    print("=" * 35)

    assistant = DeepAgentResearchAssistant()

    # Store some information
    print("Storing research insights...")
    insights = {
        "ai_trends_2024": "Multi-modal models, agent frameworks, and edge AI are key trends",
        "deep_agents_benefits": "Planning, context management, subagents, and memory persistence",
        "langchain_ecosystem": "LangChain, LangGraph, LangSmith, and DeepAgents work together"
    }

    for key, content in insights.items():
        result = assistant.store_memory_tool(key, content, "research_insights")
        print(f"✓ Stored: {key}")

    # Retrieve information
    print("\nRetrieving stored insights...")
    for key in insights.keys():
        memory = assistant.retrieve_memory_tool(key, "research_insights")
        if "error" not in memory:
            print(f"✓ {key}: {memory['content'][:50]}...")
        else:
            print(f"❌ Failed to retrieve: {key}")

    print("\n" + "=" * 35 + "\n")


async def run_all_examples():
    """Run all examples in sequence"""
    print("🚀 Deep Agent Research Assistant - Examples\n")

    # Check environment
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("❌ Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file")
        return

    try:
        await example_basic_research()
        await asyncio.sleep(2)  # Brief pause between examples

        await example_codebase_analysis()
        await asyncio.sleep(1)

        await example_memory_operations()
        await asyncio.sleep(1)

        # Uncomment these for more comprehensive testing
        # await example_intermediate_research()
        # await asyncio.sleep(2)
        # await example_advanced_research()

        print("✅ All examples completed successfully!")
        print("\n💡 Try running individual examples or create your own research topics!")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Example failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        example_functions = {
            "basic": example_basic_research,
            "intermediate": example_intermediate_research,
            "advanced": example_advanced_research,
            "codebase": example_codebase_analysis,
            "memory": example_memory_operations,
        }

        if example_name in example_functions:
            asyncio.run(example_functions[example_name]())
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available: {', '.join(example_functions.keys())}")
    else:
        asyncio.run(run_all_examples())