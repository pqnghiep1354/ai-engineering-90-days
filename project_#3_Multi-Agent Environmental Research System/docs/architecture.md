# System Architecture

## Overview

The Multi-Agent Environmental Research System uses a coordinated team of AI agents to automate research tasks. Each agent specializes in a specific function and works together through orchestrated workflows.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACES                                   │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │    CLI      │    │  Streamlit  │    │   Python    │                    │
│   │   main.py   │    │   app.py    │    │    API      │                    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
└──────────┼──────────────────┼──────────────────┼────────────────────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ResearchOrchestrator                              │   │
│   │  - Workflow selection and execution                                  │   │
│   │  - Session management                                                │   │
│   │  - History and citation tracking                                     │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WORKFLOW LAYER                                      │
│   ┌───────────────────┐              ┌───────────────────┐                  │
│   │ QuickResearch     │              │    DeepDive       │                  │
│   │   Workflow        │              │    Workflow       │                  │
│   │                   │              │                   │                  │
│   │ Research → Write  │              │ Research → Analyze│                  │
│   │                   │              │ → Research →      │                  │
│   │ (2-5 min)         │              │ Fact-Check →Write │                  │
│   │                   │              │ (10-20 min)       │                  │
│   └───────────────────┘              └───────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT LAYER                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  Researcher  │ │   Analyst    │ │    Writer    │ │ Fact-Checker │       │
│  │    Agent     │ │    Agent     │ │    Agent     │ │    Agent     │       │
│  │              │ │              │ │              │ │              │       │
│  │ • Web search │ │ • Pattern    │ │ • Report     │ │ • Verify     │       │
│  │ • Extract    │ │   analysis   │ │   structure  │ │   claims     │       │
│  │ • Summarize  │ │ • Insights   │ │ • Format     │ │ • Cross-ref  │       │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘       │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOOLS LAYER                                       │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │   Web Search    │    │    Citation     │    │   LLM Client    │        │
│   │   (Tavily)      │    │    Manager      │    │   (OpenAI)      │        │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. User Interfaces

| Interface | Description | Use Case |
|-----------|-------------|----------|
| CLI (main.py) | Command-line interface | Quick queries, scripting |
| Streamlit (app.py) | Web application | Interactive research |
| Python API | Programmatic access | Integration |

### 2. Orchestration Layer

The `ResearchOrchestrator` class manages:
- Workflow selection and instantiation
- Research session lifecycle
- History tracking
- Citation aggregation
- Report saving

### 3. Workflow Layer

#### Quick Research Workflow
```
User Query → Researcher Agent → Writer Agent → Report
```
- Duration: 2-5 minutes
- Sources: 5-10
- Best for: Simple queries, quick answers

#### Deep Dive Workflow
```
User Query → Researcher (Phase 1) 
          → Analyst 
          → Researcher (Phase 2) 
          → Fact-Checker 
          → Writer 
          → Final Report
```
- Duration: 10-20 minutes
- Sources: 15-30
- Best for: Comprehensive research, reports

### 4. Agent Layer

#### Researcher Agent
- **Purpose**: Information gathering
- **Tools**: Web Search (Tavily)
- **Output**: Sources, Findings

#### Analyst Agent
- **Purpose**: Data analysis
- **Tools**: LLM reasoning
- **Output**: Insights, Patterns

#### Writer Agent
- **Purpose**: Report generation
- **Tools**: LLM writing
- **Output**: Formatted reports

#### Fact-Checker Agent
- **Purpose**: Claim verification
- **Tools**: Web Search, Cross-reference
- **Output**: Verification status

### 5. Tools Layer

#### Web Search (Tavily)
- Real-time web search
- Configurable depth
- Domain filtering

#### Citation Manager
- Source tracking
- Multiple formats (APA, MLA, Chicago)
- Deduplication

#### LLM Client (OpenAI)
- GPT-4o-mini default
- Configurable temperature
- Tool calling support

## Data Flow

### State Management

```python
AgentState = {
    "topic": str,           # Research topic
    "messages": List,       # Conversation history
    "sources": List,        # Collected sources
    "findings": List,       # Extracted findings
    "analysis": str,        # Analysis results
    "draft": str,           # Draft report
    "final_report": str,    # Final output
    "fact_check_results": List,
    "current_agent": str,
    "iteration": int,
    "status": str,
    "errors": List,
}
```

### Agent Communication

1. State is passed between agents
2. Each agent reads relevant state
3. Agent processes and updates state
4. State flows to next agent in workflow

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
TAVILY_API_KEY=tvly-...  # For web search
OPENAI_MODEL=gpt-4o-mini
MAX_ITERATIONS=10
ENABLE_FACT_CHECKING=true
```

### Customization Points

1. **Agent Prompts**: Modify in `config.py`
2. **Workflows**: Create new workflow classes
3. **Tools**: Add new tools in `tools/`
4. **Output Formats**: Extend Writer agent

## Extensibility

### Adding New Agents

```python
from src.agents.base import BaseAgent, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            role=AgentRole.CUSTOM,
            system_prompt="Your prompt here",
        )
    
    async def process(self, state, **kwargs):
        # Implementation
        pass
```

### Adding New Workflows

```python
from src.workflows.base import BaseWorkflow

class CustomWorkflow(BaseWorkflow):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom workflow",
        )
    
    async def execute(self, topic, **kwargs):
        # Implementation
        pass
```

## Performance Considerations

| Factor | Quick Workflow | Deep Workflow |
|--------|---------------|---------------|
| API Calls | 3-5 | 10-15 |
| Time | 2-5 min | 10-20 min |
| Cost | ~$0.01-0.02 | ~$0.05-0.10 |
| Accuracy | Good | Excellent |

## Error Handling

1. **Agent Level**: Try-catch with error in response
2. **Workflow Level**: State tracks errors
3. **Orchestrator Level**: Result includes error status
4. **UI Level**: User-friendly error messages
