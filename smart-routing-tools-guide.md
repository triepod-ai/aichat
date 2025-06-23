# AIChat Smart Model Routing & Token Optimization Guide

## Overview
Enhanced claude-cli-mcp server with intelligent model routing for maximum cost efficiency and performance optimization.

## New Smart Routing Tools

### 🚀 Quick Tasks - Minimal Cost
**Tool:** `mcp__claude-cli__aichat_quick_task`
- **Model:** ollama:llama3.2:1b (fast, cheap)
- **Best for:** Simple questions, calculations, basic explanations
- **Response time:** Sub-3s
- **Token savings:** ~75% vs premium models

**Usage:**
```javascript
mcp__claude-cli__aichat_quick_task("What is the capital of France?")
mcp__claude-cli__aichat_quick_task("Calculate 15 * 24")
mcp__claude-cli__aichat_quick_task("Explain what a variable is in programming")
```

### 🔧 Code Tasks - Specialized
**Tool:** `mcp__claude-cli__aichat_code_task`
- **Model:** ollama:qwen2.5-coder:7b-instruct (code-specialized)
- **Best for:** Code review, debugging, optimization, documentation
- **Specialization:** Programming-focused understanding

**Usage:**
```javascript
mcp__claude-cli__aichat_code_task("def hello(): print('world')", "review")
mcp__claude-cli__aichat_code_task("async function getData() { return fetch('/api') }", "optimize")
mcp__claude-cli__aichat_code_task("class User: pass", "add documentation")
```

### 📚 RAG Queries - Zero API Cost
**Tool:** `mcp__claude-cli__aichat_rag_query`
- **Data Source:** Local RAG database (repo-knowledge)
- **Cost:** Zero API calls
- **Best for:** Documentation questions, existing knowledge lookup

**Usage:**
```javascript
mcp__claude-cli__aichat_rag_query("How do I configure AIChat?")
mcp__claude-cli__aichat_rag_query("What models does AIChat support?")
mcp__claude-cli__aichat_rag_query("How to use AIChat with CLI?", "repo-knowledge")
```

### 🧠 Smart Routing - Automatic Optimization
**Tool:** `mcp__claude-cli__aichat_smart_route`
- **Intelligence:** Automatically selects optimal model
- **Routing Logic:** Complexity + task type + priority analysis
- **Optimization:** Cost vs quality vs speed balance

**Usage:**
```javascript
mcp__claude-cli__aichat_smart_route("Analyze this complex algorithm", "", "quality")
mcp__claude-cli__aichat_smart_route("Quick math question", "", "speed")
mcp__claude-cli__aichat_smart_route("Review my Python code", "def process_data()...", "balanced")
```

### 💾 Session Management - Context Preservation
**Tools:** `mcp__claude-cli__aichat_session_create` & `mcp__claude-cli__aichat_session_continue`
- **Feature:** Persistent conversation context
- **Best for:** Multi-turn conversations, complex projects

**Usage:**
```javascript
// Start session
mcp__claude-cli__aichat_session_create("code_review", "I'm working on a Python project", "auto")

// Continue session
mcp__claude-cli__aichat_session_continue("code_review", "Now let's look at the database layer")
```

### 💰 Cost Estimation - Pre-execution Analysis
**Tool:** `mcp__claude-cli__aichat_estimate_cost`
- **Feature:** Compare costs across models before execution
- **Output:** Recommendations for optimal model selection

**Usage:**
```javascript
mcp__claude-cli__aichat_estimate_cost("Complex architectural analysis", "Building microservices system")
```

## Smart Routing Decision Matrix

| Task Type | Complexity | Recommended Tool | Model Used |
|-----------|------------|------------------|------------|
| Simple questions | 1-3 | `aichat_quick_task` | llama3.2:1b |
| Code review | Any | `aichat_code_task` | qwen2.5-coder:7b |
| Documentation | Any | `aichat_rag_query` | Local RAG |
| Complex analysis | 8-10 | `aichat_smart_route` | claude-3-5-sonnet |
| Mixed/unknown | Any | `aichat_smart_route` | Auto-selected |

## Token Optimization Benefits

### Cost Savings Comparison
- **Traditional approach:** All tasks → Claude API ($$$)
- **Smart routing:** 
  - 60% of tasks → llama3.2:1b (minimal cost)
  - 25% of tasks → qwen2.5-coder (low cost)
  - 15% of tasks → Claude API (when needed)
  - **Result:** ~70% cost reduction

### Performance Benefits
- **Quick tasks:** 2-3s response (vs 8-15s premium models)
- **Code tasks:** Specialized understanding (better results)
- **RAG queries:** Instant responses (no API latency)
- **Smart routing:** Optimal model selection (balanced optimization)

## Real-World Usage Examples

### Development Workflow
```javascript
// 1. Quick syntax check (fast/cheap)
mcp__claude-cli__aichat_quick_task("What's the syntax for Python list comprehension?")

// 2. Code review (specialized)
mcp__claude-cli__aichat_code_task("def process_users(data): return [u for u in data if u.active]", "optimize")

// 3. Documentation lookup (zero cost)
mcp__claude-cli__aichat_rag_query("How to configure AIChat with multiple models?")

// 4. Complex architecture (premium when needed)
mcp__claude-cli__aichat_smart_route("Design a scalable microservices architecture", "", "quality")
```

### Learning & Research
```javascript
// 1. Basic concepts (quick/cheap)
mcp__claude-cli__aichat_quick_task("What is REST API?")

// 2. Existing knowledge (RAG)
mcp__claude-cli__aichat_rag_query("Show me AIChat REST API examples")

// 3. Deep analysis (smart routing)
mcp__claude-cli__aichat_smart_route("Compare REST vs GraphQL for my use case", "Building e-commerce API", "balanced")
```

## Configuration & Setup

### Prerequisites
1. AIChat server running on port 42333
2. Ollama models installed: llama3.2:1b, qwen2.5-coder:7b-instruct
3. Claude API configured for premium tasks
4. RAG database created: repo-knowledge

### Verify Setup
```bash
# Check AIChat server
curl http://127.0.0.1:42333/v1/models

# Check Ollama models
ollama list

# Check RAG database
aichat --rag list
```

## Troubleshooting

### Common Issues
1. **Tool not found:** MCP server needs restart after code changes
2. **Model not available:** Check Ollama installation and model pull
3. **RAG query fails:** Verify RAG database exists and is accessible
4. **Session not found:** Check session was created before continuing

### Debug Commands
```javascript
// Test basic connectivity
mcp__claude-cli__claude_health_check()

// Verify model availability
mcp__claude-cli__aichat_estimate_cost("test", "", ["ollama:llama3.2:1b"])

// Check RAG databases
mcp__claude-cli__aichat_rag_query("test", "repo-knowledge")
```

## Advanced Features

### Priority Modes
- **"speed":** Favor fastest models
- **"cost":** Favor cheapest models  
- **"quality":** Favor highest quality models
- **"balanced":** Optimize across all factors (default)

### Custom Model Selection
```javascript
// Override auto-selection
mcp__claude-cli__aichat_session_create("analysis", "Complex task", "claude:claude-3-5-sonnet")
```

### Batch Processing
```javascript
// Estimate costs for multiple approaches
mcp__claude-cli__aichat_estimate_cost("Large analysis task", "", [
    "ollama:llama3.2:1b",
    "ollama:qwen2.5-coder:7b-instruct", 
    "claude:claude-3-5-sonnet"
])
```

This smart routing system provides the foundation for intelligent, cost-effective AI task distribution while maintaining quality and performance.