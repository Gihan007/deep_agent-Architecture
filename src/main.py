#!/usr/bin/env python3
"""
Deep Agent Research Assistant - Main Application

This application demonstrates all core Deep Agents concepts:
1. Planning and task decomposition (write_todos)
2. Context management with file system tools
3. Subagent spawning for specialized tasks
4. Long-term memory persistence
5. Custom tools and middleware
"""

# Standard libraries
import asyncio
import json
import operator
import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# Third-party 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
import requests
from tavily import TavilyClient

# Deep Agents import
from deepagents import create_deep_agent

# Local imports
# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Define graph state for custom architectures
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    context: Dict[str, Any]
    memory: Dict[str, Any]
    current_task: str
    completed_tasks: List[str]


class DeepAgentResearchAssistant:
    """Main class for the Deep Agent Research Assistant with multiple graph architectures"""

    def __init__(self, graph_architecture="default"):
        
        # Initialize API clients
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # Set OpenAI API key as environment variable for LangChain
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4")
        print("✅ API clients initialized")

        # Initialize memory store
        self.store = InMemoryStore()

        # Create workspace directory for file operations
        self.workspace_dir = Path("workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        # Store graph architecture choice
        self.graph_architecture = graph_architecture

        # System prompt incorporating all concepts
        self.system_prompt = """
        You are an expert research assistant powered by Deep Agents. You excel at complex, multi-step research tasks.

        ## Core Capabilities:
        1. **Planning & Task Decomposition**: Use `write_todos` to break down complex research into manageable steps
        2. **Context Management**: Use file system tools to manage large amounts of information
        3. **Subagent Spawning**: Use the `task` tool to delegate specialized subtasks to subagents
        4. **Long-term Memory**: Store and retrieve important findings across conversations

        ## Research Process:
        - For complex research questions: Start by planning your approach with `write_todos`
        - Use `internet_search` when you need external information or current data
        - For simple questions you can answer directly, provide the answer without using tools
        - Save large results to files to manage context when dealing with extensive information
        - Spawn subagents for deep dives into specific areas when the task is complex
        - Store key insights in memory for future reference
        - Synthesize findings into comprehensive reports

        ## When to use tools:
        - Use `internet_search` for: current events, specific facts, recent developments, or when you need to verify information
        - Use `write_todos` for: multi-step tasks, complex analysis, or when breaking down research into phases
        - Use file tools for: managing large amounts of information or preserving context
        - Use memory tools for: storing important findings or retrieving previous knowledge
        """

        # Custom tools
        self.custom_tools = [
            self.internet_search_tool,
            self.analyze_codebase_tool,
            self.store_memory_tool,
            self.retrieve_memory_tool,
            self.write_todos_tool,
            self.task_delegation_tool,
        ]

        # Create the deep agent with chosen architecture
        self.agent = self._create_agent_with_architecture()

    def _create_agent_with_architecture(self):
        """Create agent with different graph architectures"""
        if self.graph_architecture == "hierarchical":
            return self._create_hierarchical_agent()
        elif self.graph_architecture == "parallel":
            return self._create_parallel_agent()
        elif self.graph_architecture == "workflow":
            return self._create_workflow_agent()
        else:  # default
            return self._create_default_agent()

    def _create_default_agent(self):
        """Create the default Deep Agent using the deepagents library"""
        return create_deep_agent(
            tools=self.custom_tools,
            system_prompt=self.system_prompt,
            store=self.store,
        )
    
    #This is the Entire Sub Architectures- Hierarchical Agents Architecture
    def _create_hierarchical_agent(self):
        """Create a hierarchical agent that uses tool-enabled phases"""
        # Create tools as standalone functions that capture self
        from functools import partial
        
        def internet_search(query: str, max_results: int = 5, topic: str = "general", include_raw_content: bool = False) -> Dict[str, Any]:
            """Search the internet using Tavily (primary) or DuckDuckGo (free fallback)"""
            return self.internet_search_tool.func(self, query, max_results, topic, include_raw_content)
        
        def analyze_codebase(repo_url: str, focus_areas: List[str] = None) -> Dict[str, Any]:
            """Analyze a codebase for specific focus areas"""
            return self.analyze_codebase_tool.func(self, repo_url, focus_areas)
        
        def store_memory(key: str, content: str, namespace: str = "research") -> str:
            """Store information in long-term memory"""
            return self.store_memory_tool.func(self, key, content, namespace)
        
        def retrieve_memory(key: str, namespace: str = "research") -> Dict[str, Any]:
            """Retrieve information from long-term memory"""
            return self.retrieve_memory_tool.func(self, key, namespace)
        
        def write_todos(task_description: str, subtasks: List[str]) -> str:
            """Create a structured task decomposition plan"""
            return self.write_todos_tool.func(self, task_description, subtasks)
        
        def task_delegation(subtask_description: str, agent_type: str = "specialist") -> str:
            """Delegate a subtask to a specialized analysis"""
            return self.task_delegation_tool.func(self, subtask_description, agent_type)
        
        tools = [
            tool(internet_search),
            tool(analyze_codebase),
            tool(store_memory),
            tool(retrieve_memory),
            tool(write_todos),
            tool(task_delegation),
        ]
        
        # Use create_react_agent with phase-specific prompts
        
        planning_prompt = """
        You are in the planning phase of research. Your task is to:
        1. Break down the research topic into manageable subtasks
        2. Use write_todos to create a structured plan
        3. Focus only on planning, do not execute research yet
        """
        
        research_prompt = """
        You are in the research execution phase. Your task is to:
        1. Execute the research plan created in planning
        2. Use internet_search to gather information
        3. Use task_delegation for specialized analysis
        4. Save findings to files for context management
        5. Store key insights in memory
        """
        
        synthesis_prompt = """
        You are in the synthesis phase. Your task is to:
        1. Review all findings from the research phase
        2. Combine information into a comprehensive report
        3. Provide clear, well-structured conclusions
        """
        
        # Create agents for each phase
        planner_agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=planning_prompt,
        )
        
        researcher_agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=research_prompt,
        )
        
        synthesizer_agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=synthesis_prompt,
        )

        def planner_node(state: AgentState) -> AgentState:
            """Planning node that decomposes tasks"""
            messages = state["messages"]
            result = planner_agent.invoke({"messages": messages})
            state["current_task"] = "planning"
            state["messages"] = messages + [result["messages"][-1]]
            return state

        def research_node(state: AgentState) -> AgentState:
            """Research execution node"""
            messages = state["messages"]
            result = researcher_agent.invoke({"messages": messages})
            state["current_task"] = "researching"
            state["messages"] = messages + [result["messages"][-1]]
            return state

        def synthesis_node(state: AgentState) -> AgentState:
            """Synthesis node that combines findings"""
            messages = state["messages"]
            result = synthesizer_agent.invoke({"messages": messages})
            state["current_task"] = "synthesizing"
            state["messages"] = messages + [result["messages"][-1]]
            return state

        # Create hierarchical graph
        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_node)
        graph.add_node("researcher", research_node)
        graph.add_node("synthesizer", synthesis_node)

        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "synthesizer")
        graph.add_edge("synthesizer", END)

        graph.set_entry_point("planner")

        # Visualize the graph
        compiled_graph = graph.compile()
        # Increase recursion limit for complex hierarchical research
        compiled_graph = compiled_graph.with_config({"recursion_limit": 50})
        graph_image = compiled_graph.get_graph().draw_mermaid_png()
        with open(self.workspace_dir / "hierarchical_graph.png", "wb") as f:
            f.write(graph_image)
        
        # Save Mermaid text for reference
        mermaid_text = compiled_graph.get_graph().draw_mermaid()
        with open(self.workspace_dir / "hierarchical_graph.mmd", "w") as f:
            f.write(mermaid_text)

        return compiled_graph
    
    #This is the Entire Sub Architectures- parallel Agents Architecture
    def _create_parallel_agent(self):
        """Create a parallel processing agent using Deep Agents"""
        # Use create_deep_agent with parallel-focused prompt
        parallel_prompt = """
        You are a parallel processing research agent using Deep Agents. Your task is to:
        1. Break down the research topic into multiple parallel subtasks
        2. Use write_todos to plan parallel research streams
        3. Execute multiple research approaches simultaneously using available tools
        4. Use internet_search for gathering information from different sources
        5. Spawn multiple subagents for specialized parallel analysis
        6. Save findings to files and store insights in memory
        7. Synthesize parallel findings into comprehensive results
        
        Focus on parallel processing - handle multiple aspects simultaneously rather than sequentially.
        Use Deep Agents subagent spawning capabilities for parallel execution.
        """
        
        return create_deep_agent(
            tools=self.custom_tools,
            system_prompt=parallel_prompt,
            store=self.store,
        )

    def _create_workflow_agent(self):
        """Create a workflow-based agent with conditional logic"""
        def router_node(state: AgentState) -> str:
            """Route based on task complexity"""
            task = state.get("current_task", "")
            if "complex" in task.lower():
                return "hierarchical"
            elif "simple" in task.lower():
                return "direct"
            else:
                return "standard"

        def direct_response_node(state: AgentState) -> AgentState:
            """Handle simple queries directly"""
            messages = state["messages"]
            response = self.llm.invoke(messages)
            state["messages"] = messages + [response]
            return state

        graph = StateGraph(AgentState)
        graph.add_node("router", router_node)
        graph.add_node("direct", direct_response_node)
        graph.add_node("hierarchical", self._create_hierarchical_agent())
        graph.add_node("standard", self._create_default_agent())

        graph.add_conditional_edges("router", lambda x: x, {
            "hierarchical": "hierarchical",
            "direct": "direct",
            "standard": "standard"
        })

        graph.set_entry_point("router")

        # Visualize the graph (note: sub-agents may not render fully)
        try:
            compiled_graph = graph.compile()
            graph_image = compiled_graph.get_graph().draw_mermaid_png()
            with open(self.workspace_dir / "workflow_graph.png", "wb") as f:
                f.write(graph_image)
        except Exception as e:
            print(f"Warning: Could not visualize workflow graph: {e}")

        return compiled_graph

    def duckduckgo_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Free web search using DuckDuckGo Instant Answer API"""
        try:
            url = f'https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1'
            response = requests.get(url, timeout=10)
            data = response.json()

            results = []

            # Add abstract if available
            if 'AbstractText' in data and data['AbstractText']:
                results.append({
                    'title': data.get('Heading', 'Abstract'),
                    'content': data['AbstractText'],
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Abstract'
                })

            # Add related topics
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics']:
                    if 'Text' in topic and len(results) < max_results:
                        results.append({
                            'title': topic.get('FirstURL', 'Related Topic'),
                            'content': topic['Text'][:300],  # Limit content length
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo Related'
                        })

            # Add answer if available
            if 'Answer' in data and data['Answer'] and len(results) < max_results:
                results.append({
                    'title': 'Direct Answer',
                    'content': data['Answer'],
                    'url': data.get('AnswerURL', ''),
                    'source': 'DuckDuckGo Answer'
                })

            return {'results': results[:max_results]}

        except Exception as e:
            return {'error': f'DuckDuckGo search failed: {str(e)}', 'results': []}

    @tool
    def internet_search_tool(
        self,
        query: str,
        max_results: int = 5,
        topic: str = "general",
        include_raw_content: bool = False
    ) -> Dict[str, Any]:
        """Search the internet using Tavily (primary) or DuckDuckGo (free fallback)"""
        # Try Tavily first
        try:
            results = self.tavily_client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                include_raw_content=include_raw_content
            )
            # Add source indicator
            if 'results' in results:
                for result in results['results']:
                    result['source'] = 'Tavily'
            return results
        except Exception as tavily_error:
            print(f"🔄 Tavily search failed, falling back to free DuckDuckGo search...")
            try:
                # Fall back to free DuckDuckGo search
                duck_results = self.duckduckgo_search(query, max_results)
                # Ensure it has the expected format
                if 'results' not in duck_results:
                    duck_results = {'results': [], 'error': 'No results from DuckDuckGo'}
                return duck_results
            except Exception as duck_error:
                # If both fail, return a safe fallback
                return {
                    'results': [{
                        'title': f'Search failed for: {query}',
                        'content': f'Both Tavily and DuckDuckGo searches failed. Tavily: {str(tavily_error)[:100]}, DuckDuckGo: {str(duck_error)[:100]}',
                        'url': '',
                        'source': 'Error'
                    }],
                    'error': f'Search failed: {str(tavily_error)}'
                }

    @tool
    def analyze_codebase_tool(self, repo_url: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Analyze a codebase for specific focus areas"""
        if focus_areas is None:
            focus_areas = ["architecture", "dependencies", "key_features"]

        # This would integrate with GitHub API or similar
        # For demo purposes, return mock analysis
        return {
            "repo_url": repo_url,
            "analysis": {
                "architecture": "Well-structured modular design",
                "dependencies": ["langchain", "deepagents", "anthropic"],
                "key_features": ["Planning", "Context management", "Subagents"],
                "complexity": "Intermediate",
                "recommendations": ["Add more error handling", "Implement caching"]
            }
        }

    @tool
    def store_memory_tool(self, key: str, content: str, namespace: str = "research") -> str:
        """Store information in long-term memory"""
        try:
            self.store.put((namespace, key), {"content": content, "timestamp": str(asyncio.get_event_loop().time())})
            return f"Successfully stored memory with key: {key}"
        except Exception as e:
            return f"Failed to store memory: {str(e)}"

    @tool
    def retrieve_memory_tool(self, key: str, namespace: str = "research") -> Dict[str, Any]:
        """Retrieve information from long-term memory"""
        try:
            memory = self.store.get((namespace, key))
            if memory:
                return memory.value
            else:
                return {"error": f"No memory found for key: {key}"}
        except Exception as e:
            return {"error": f"Failed to retrieve memory: {str(e)}"}

    @tool
    def write_todos_tool(self, task_description: str, subtasks: List[str]) -> str:
        """Create a structured task decomposition plan"""
        try:
            todo_content = f"""
# Task Plan: {task_description}

## Subtasks:
{chr(10).join(f"{i+1}. {task}" for i, task in enumerate(subtasks))}

## Status: Planning Phase
"""
            # Save to workspace
            todo_file = self.workspace_dir / f"todo_{task_description.replace(' ', '_')[:50]}.md"
            todo_file.write_text(todo_content)

            return f"Successfully created task plan with {len(subtasks)} subtasks. Saved to {todo_file.name}"
        except Exception as e:
            return f"Failed to create task plan: {str(e)}"

    @tool
    def task_delegation_tool(self, subtask_description: str, agent_type: str = "specialist") -> str:
        """Delegate a subtask to a specialized subagent"""
        try:
            # Create a subagent prompt
            subagent_prompt = f"""
You are a specialized {agent_type} subagent. Focus on: {subtask_description}

Provide detailed, expert-level analysis on this specific aspect.
Be thorough but concise. Use available tools when needed.
"""

            # In a real implementation, this would spawn an actual subagent
            # For demo purposes, we'll simulate subagent work
            subagent_file = self.workspace_dir / f"subagent_{agent_type}_{subtask_description.replace(' ', '_')[:30]}.md"

            # Simulate subagent work by calling the LLM
            response = self.llm.invoke([HumanMessage(content=subagent_prompt)])

            subagent_content = f"""
# Subagent Report: {agent_type}
## Task: {subtask_description}

{response.content}

## Status: Completed by {agent_type} subagent
"""
            subagent_file.write_text(subagent_content)

            return f"Subagent ({agent_type}) completed task. Results saved to {subagent_file.name}"
        except Exception as e:
            return f"Subagent delegation failed: {str(e)}"

    async def research_topic(self, topic: str, depth: str = "intermediate") -> Dict[str, Any]:
        """Conduct comprehensive research on a topic using the chosen graph architecture"""
        prompt = f"""
        Conduct a comprehensive {depth}-level research on: {topic}

        Follow this process:
        1. Plan your research approach using write_todos
        2. Gather information using internet_search
        3. Save detailed findings to files for context management
        4. Spawn subagents for specialized analysis (e.g., technical details, history, current trends)
        5. Store key insights in memory for future reference
        6. Synthesize everything into a final report

        Be thorough but efficient. Use file system tools to manage large amounts of information.

        Graph Architecture: {self.graph_architecture}
        """

        try:
            if self.graph_architecture == "default":
                # Use standard DeepAgents interface
                result = await self.agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
                return {
                    "success": True,
                    "response": result["messages"][-1].content,
                    "full_result": result
                }
            else:
                # Use custom graph with state
                initial_state = AgentState(
                    messages=[HumanMessage(content=prompt)],
                    context={},
                    memory={},
                    current_task="research",
                    completed_tasks=[]
                )

                result = self.agent.invoke(initial_state)
                return {
                    "success": True,
                    "response": result["messages"][-1].content if result["messages"] else "Research completed",
                    "full_result": result,
                    "architecture": self.graph_architecture
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def list_workspace_files(self) -> List[str]:
        """List files in the workspace directory"""
        return [str(f.relative_to(self.workspace_dir)) for f in self.workspace_dir.rglob("*") if f.is_file()]

    def get_file_content(self, filename: str) -> Optional[str]:
        """Get content of a workspace file"""
        file_path = self.workspace_dir / filename
        if file_path.exists():
            return file_path.read_text()
        return None


def main():
    """
    Main CLI entry point
    """
    # python your_script.py "Quantum Computing"
    # python your_script.py "AI Ethics" --depth basic
    # python your_script.py "Climate Change" --graph hierarchical
    # python your_script.py "Renewable Energy Trends" --depth advanced --graph parallel
    # python your_script.py "Machine Learning" --depth intermediate --graph workflow --list-files
    # python your_script.py "Data Science" --read-file data.csv
    # python your_script.py "Blockchain Technology" --depth advanced --graph parallel --list-files --read-file report.md
    import argparse

    parser = argparse.ArgumentParser(description="Deep Agent Research Assistant")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--depth", choices=["basic", "intermediate", "advanced"], default="intermediate",
                       help="Research depth level")
    parser.add_argument("--graph", choices=["default", "hierarchical", "parallel", "workflow"],
                       default="default", help="Graph architecture to use")
    parser.add_argument("--list-files", action="store_true", help="List workspace files")
    parser.add_argument("--read-file", help="Read content of a workspace file")

    args = parser.parse_args()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Check required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY environment variable not set")
        return
    # Note: Deep Agents can work with OpenAI only - Anthropic is optional for subagents

    print(f"🤖 Initializing Deep Agent with {args.graph} architecture...")
    assistant = DeepAgentResearchAssistant(graph_architecture=args.graph)

    if args.list_files:
        files = assistant.list_workspace_files()
        print("Workspace files:")
        for file in files:
            print(f"  - {file}")
        return

    if args.read_file:
        content = assistant.get_file_content(args.read_file)
        if content:
            print(f"Content of {args.read_file}:")
            print(content)
        else:
            print(f"File {args.read_file} not found")
        return

    # Run research
    print(f"Starting {args.depth} research on: {args.topic}")
    print("This may take a few minutes...")

    async def run_research():
        result = await assistant.research_topic(args.topic, args.depth)
        if result["success"]:
            print("\n" + "="*50)
            print("RESEARCH RESULTS")
            print("="*50)
            print(result["response"])
        else:
            print(f"Research failed: {result['error']}")

    asyncio.run(run_research())


if __name__ == "__main__":
    main()