# Agent Design Documentation

## Overview

This document describes the design principles and implementation details for each agent in the Multi-Agent Environmental Research System.

## Base Agent Architecture

All agents inherit from `BaseAgent` and implement a consistent interface:

```python
class BaseAgent(ABC):
    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        model_name: str,
        temperature: float,
        tools: List[Any],
    ):
        pass
    
    @abstractmethod
    async def process(
        self,
        state: AgentState,
        **kwargs,
    ) -> AgentResponse:
        pass
```

## Agent Specifications

### 1. Researcher Agent

**Purpose**: Gather information from web sources

**System Prompt Focus**:
- Prioritize authoritative sources (IPCC, EPA, UN)
- Extract specific facts and data
- Maintain citation accuracy
- Flag conflicting information

**Process Flow**:
```
1. Generate search queries from topic
2. Execute web searches
3. Deduplicate sources
4. Extract findings with citations
5. Generate summary
```

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.3 | Low for factual accuracy |
| max_sources | 10 | Maximum sources to collect |

**Output**:
- `sources`: List of Source objects
- `findings`: List of Finding objects
- `content`: Summary text

---

### 2. Analyst Agent

**Purpose**: Analyze research data and generate insights

**System Prompt Focus**:
- Identify patterns and trends
- Provide balanced analysis
- Support conclusions with evidence
- Acknowledge limitations

**Analysis Types**:

1. **Comprehensive Analysis**
   - Key themes
   - Evidence assessment
   - Stakeholder perspectives
   - Gaps and limitations
   - Implications
   - Recommendations

2. **Trend Analysis**
   - Historical context
   - Current state
   - Emerging trends
   - Future projections
   - Driving factors

3. **Comparative Analysis**
   - Key dimensions
   - Similarities/Differences
   - Best practices
   - Trade-offs

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.4 | Balanced for reasoning |
| analysis_type | comprehensive | Analysis approach |

**Output**:
- `content`: Analysis text
- `findings`: Extracted insights

---

### 3. Writer Agent

**Purpose**: Generate well-structured research reports

**System Prompt Focus**:
- Clear, professional prose
- Proper citation usage
- Logical structure
- Appropriate formatting

**Report Formats**:

1. **Full Report**
   - Executive Summary
   - Introduction
   - Methodology
   - Key Findings
   - Analysis and Discussion
   - Conclusions and Recommendations
   - References

2. **Executive Summary**
   - 300-500 words
   - Key conclusions first
   - Bullet points for takeaways

3. **Brief**
   - 150-250 words
   - Essential points only

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.5 | Creative but coherent |
| report_format | full | Output format |
| include_citations | true | Add references |

**Output**:
- `content`: Formatted report

---

### 4. Fact-Checker Agent

**Purpose**: Verify claims and ensure accuracy

**System Prompt Focus**:
- Rigorous verification
- Multiple source cross-reference
- Clear uncertainty flagging
- Prioritize accuracy

**Verification Process**:
```
1. Extract verifiable claims
2. Search for verification
3. Analyze source consistency
4. Assign confidence levels
5. Generate verification report
```

**Claim Categories**:
- Statistical claims (numbers, dates)
- Causal claims (X causes Y)
- Comparative claims (A > B)
- Attribution claims (X said Y)

**Verification Statuses**:
| Status | Confidence | Description |
|--------|------------|-------------|
| VERIFIED | 80-100% | Multiple sources confirm |
| PARTIALLY VERIFIED | 50-80% | Some support found |
| UNVERIFIED | 20-50% | Cannot confirm |
| FALSE | 0-20% | Sources contradict |

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.2 | Very low for accuracy |
| max_claims | 10 | Claims to verify |

**Output**:
- `content`: Verification summary
- `findings`: Verified/flagged claims
- `metadata.details`: Full verification results

---

## Agent Communication

### State Passing

Agents communicate through shared `AgentState`:

```python
# Researcher updates state
state["sources"].extend(new_sources)
state["findings"].extend(new_findings)

# Analyst reads findings
findings = state["findings"]
analysis = await self.analyze(findings)
state["analysis"] = analysis

# Writer uses all state
report = await self.write(
    topic=state["topic"],
    findings=state["findings"],
    analysis=state["analysis"],
)
state["final_report"] = report
```

### Message History

Each agent adds to message history:

```python
state["messages"].append(
    AIMessage(content=response.content, name=agent_role)
)
```

## Error Handling

### Agent-Level Errors

```python
try:
    result = await self._call_llm(messages)
except Exception as e:
    return AgentResponse(
        agent=self.role,
        content="",
        success=False,
        error=str(e),
    )
```

### Graceful Degradation

- If Researcher fails → Return partial results
- If Analyst fails → Skip to Writer
- If Fact-Checker fails → Proceed without verification
- If Writer fails → Return raw findings

## Best Practices

### 1. Prompt Engineering

- Be specific about output format
- Include examples when helpful
- Set clear constraints
- Define success criteria

### 2. Temperature Settings

| Task Type | Temperature |
|-----------|-------------|
| Fact extraction | 0.2-0.3 |
| Analysis | 0.4-0.5 |
| Creative writing | 0.5-0.7 |
| Brainstorming | 0.7-0.9 |

### 3. Context Management

- Keep prompts focused
- Summarize long contexts
- Prioritize recent/relevant info
- Truncate when necessary

### 4. Cost Optimization

- Use gpt-4o-mini for most tasks
- Batch similar operations
- Cache repeated queries
- Limit max_tokens appropriately
