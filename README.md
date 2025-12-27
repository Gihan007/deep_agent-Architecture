# Deep Agent Research Assistant

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Gihan007/deep_agent-Architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intermediate-level project demonstrating all core concepts of LangChain's Deep Agents framework. This application showcases planning, task decomposition, context management, subagent spawning, and long-term memory in a comprehensive research assistant.

## 🚀 Features

### Core Deep Agents Concepts Demonstrated

1. **Planning & Task Decomposition**
   - Uses built-in `write_todos` tool to break down complex research tasks
   - Automatically plans research approaches based on user queries

2. **Context Management**
   - File system tools (`ls`, `read_file`, `write_file`, `edit_file`) for managing large amounts of information
   - Offloads search results and analysis to disk to prevent context window overflow

3. **Subagent Spawning**
   - `task` tool enables delegation of specialized subtasks to subagents
   - Context isolation for different research aspects (technical, historical, current trends)

4. **Long-term Memory**
   - Persistent memory across conversations using LangGraph Store
   - Store and retrieve key insights for future research

5. **Custom Tools**
   - Internet search integration (Tavily)
   - Codebase analysis capabilities
   - Memory management tools

6. **Graph Architectures & Visualization**
   - Multiple agent graph architectures: default, hierarchical, parallel, workflow
   - Automatic graph visualization saved as PNG files in the workspace directory
   - Uses Mermaid diagrams for clear representation of agent flows

## 📁 Project Structure

```
deep_agent-Architecture/
├── src/
│   ├── __init__.py
│   └── main.py                 # Main application
├── docs/
│   └── article.md              # Technical article about Deep Agents
├── examples/
│   ├── todo_AI_Ethics_Research.md
│   └── analysis_report.md      # Sample research outputs
├── workspace/                  # Generated files (gitignored)
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## 🛠️ Quick Start

### 1. **Clone the repository**
```bash
git clone https://github.com/Gihan007/deep_agent-Architecture.git
cd deep_agent-Architecture
```

### 2. **Set up API keys**:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys (get them from the links below)
# OPENAI_API_KEY: https://platform.openai.com/api-keys
# TAVILY_API_KEY: https://tavily.com/
```

### 3. **Install dependencies**:
```bash
pip install -e .
```

### 4. **Run the research assistant**:
```bash
# Basic research
python src/main.py "AI Ethics"

# Advanced research with parallel processing
python src/main.py "Machine Learning" --graph parallel --depth advanced

# List generated workspace files
python src/main.py "Any Topic" --list-files
```

## 🎯 Usage Examples

### Command Line Interface
```bash
# Different research depths
python src/main.py "Quantum Computing" --depth basic
python src/main.py "Climate Change" --depth intermediate
python src/main.py "AI Safety" --depth advanced

# Different agent architectures
python src/main.py "Blockchain" --graph default      # Simple agent
python src/main.py "IoT Security" --graph hierarchical  # Sequential phases
python src/main.py "5G Networks" --graph parallel    # Concurrent subagents
python src/main.py "Edge Computing" --graph workflow # Conditional routing
```

### Programmatic Usage
```python
from src.main import DeepAgentResearchAssistant

# Initialize the assistant
assistant = DeepAgentResearchAssistant(graph_architecture="parallel")

# Run research
result = await assistant.research_topic("Your Research Topic", "intermediate")
print(result["response"])
```

## 📚 Documentation

- **[Technical Article](docs/article.md)**: Comprehensive guide to Deep Agents framework
- **[API Reference](src/main.py)**: Complete code documentation
- **[Examples](examples/)**: Sample research outputs and task plans

## 🔧 Architecture Details

### Agent Types
- **Default Agent**: Simple `create_deep_agent()` implementation
- **Hierarchical Agent**: Sequential planner → researcher → synthesizer phases
- **Parallel Agent**: Concurrent subagent spawning with coordination
- **Workflow Agent**: Conditional routing based on task complexity

### Tool Ecosystem
- **Internet Search**: Tavily API with DuckDuckGo fallback
- **Task Planning**: Automated research decomposition
- **Memory Management**: Persistent context storage
- **File Operations**: Autonomous document management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the Deep Agents framework
- **OpenAI** for GPT-4 API
- **Tavily** for intelligent web search
- **Anthropic** for Claude API (optional subagent enhancement)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Gihan007/deep_agent-Architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Gihan007/deep_agent-Architecture/discussions)

---

**Star this repository** ⭐ if you find it helpful for learning Deep Agents!

#DeepAgents #LangChain #AI #MultiAgentSystems #MachineLearning #Python

3. **Run tests**:
   ```bash
   python test.py
   ```

4. **Try the examples**:
   ```bash
   # Run all examples
   python examples.py

   # Or run a specific example
   python examples.py basic
   ```

5. **Start researching**:
   ```bash
   # Basic research
   python -m src.main "What is machine learning?"

   # Advanced research with planning
   python -m src.main "Compare different LLM architectures" --depth advanced
   ```

## 🔧 Configuration

### Required API Keys

- **ANTHROPIC_API_KEY**: Get from [Anthropic Console](https://console.anthropic.com/)
- **TAVILY_API_KEY**: Get from [Tavily](https://tavily.com/)

### Optional Configuration

- **LANGSMITH_API_KEY**: For observability and debugging
- **OPENAI_API_KEY**: Alternative LLM provider

## 📖 Usage

### Graph Architectures

The assistant supports different agent graph architectures:

- **default**: Standard Deep Agents behavior
- **hierarchical**: Planner → Researcher → Synthesizer flow
- **parallel**: Parallel research execution
- **workflow**: Conditional routing based on task complexity

To use a specific architecture:

```python
from src.main import DeepAgentResearchAssistant

# Hierarchical architecture
assistant = DeepAgentResearchAssistant(graph_architecture="hierarchical")
```

Graph visualizations are automatically saved as PNG files in the `workspace/` directory.

### Basic Research

```bash
# Research a topic with default intermediate depth
python -m src.main "What is LangChain Deep Agents?"

# Specify research depth
python -m src.main "Neural Networks" --depth advanced

# Basic research
python -m src.main "Python async programming" --depth basic
```

### Workspace Management

```bash
# List files created by the agent
python -m src.main --list-files "dummy_topic"

# Read a specific file
python -m src.main --read-file research_findings.txt "dummy_topic"
```

### Programmatic Usage

```python
from src.main import DeepAgentResearchAssistant

assistant = DeepAgentResearchAssistant()

# Asynchronous research
import asyncio

async def research():
    result = await assistant.research_topic("Your research topic")
    print(result["response"])

asyncio.run(research())
```

## 🏗️ Project Structure

```
deep-agent-research-assistant/
├── src/
│   ├── __init__.py
│   └── main.py              # Main application logic
├── workspace/               # Generated files and context
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔍 How It Works

### Research Process

1. **Planning Phase**: Agent uses `write_todos` to decompose the research task
2. **Information Gathering**: Uses `internet_search` tool to collect data
3. **Context Management**: Saves large results to files using file system tools
4. **Specialized Analysis**: Spawns subagents for deep dives into specific areas
5. **Memory Storage**: Stores key insights in long-term memory
6. **Synthesis**: Compiles findings into a comprehensive report

### Example Agent Behavior

When asked to research "LangChain Deep Agents":

1. **Planning**: Creates todos for overview, capabilities, use cases, examples
2. **Search**: Queries web for official docs, examples, comparisons
3. **File Management**: Saves search results to `research_data.json`
4. **Subagents**: Spawns agents for "technical details" and "comparison analysis"
5. **Memory**: Stores key concepts like "planning", "context management", etc.
6. **Report**: Generates structured summary with all findings

## 🧪 Testing the Concepts

### 1. Planning & Decomposition
```bash
python -m src.main "Research the history and future of quantum computing"
```
Watch the agent create and manage a todo list.

### 2. Context Management
```bash
python -m src.main "Analyze the entire Python ecosystem" --depth advanced
```
See how large amounts of information are saved to files.

### 3. Subagent Spawning
```bash
python -m src.main "Compare React vs Vue vs Angular frameworks"
```
Observe subagents being spawned for each framework analysis.

### 4. Long-term Memory
```bash
# First research
python -m src.main "What are microservices?"

# Later research that builds on previous knowledge
python -m src.main "How do microservices relate to serverless architecture?"
```
The agent will recall previous findings.

## 🔧 Customization

### Adding New Tools

Extend the `custom_tools` list in `DeepAgentResearchAssistant.__init__()`:

```python
@tool
def your_custom_tool(self, param: str) -> str:
    # Your tool logic here
    return result

self.custom_tools.append(self.your_custom_tool)
```

### Modifying System Prompt

Update `self.system_prompt` to change agent behavior and capabilities.

### Changing LLM Provider

Replace `ChatAnthropic` with other LangChain chat models in `__init__()`.

## 📊 Observability

Enable LangSmith for detailed tracing:

1. Set `LANGSMITH_API_KEY` in `.env`
2. Set `LANGSMITH_PROJECT` to your project name
3. View traces at [LangSmith](https://smith.langchain.com/)

## 🚀 Deployment

### Local Development
```bash
# Run with auto-reload
pip install watchfiles
watchfiles "python -m src.main 'test topic'" src/
```

### Production Deployment
Use LangSmith Deployment or deploy as a web service:

```python
from fastapi import FastAPI
from src.main import DeepAgentResearchAssistant

app = FastAPI()
assistant = DeepAgentResearchAssistant()

@app.post("/research")
async def research_endpoint(topic: str, depth: str = "intermediate"):
    result = await assistant.research_topic(topic, depth)
    return result
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 Learn More

- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents)
- [LangGraph Guide](https://docs.langchain.com/oss/python/langgraph)
- [LangChain Tools](https://docs.langchain.com/oss/python/langchain/tools)

## 📄 License

MIT License - see LICENSE file for details
