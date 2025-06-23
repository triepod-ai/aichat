# AIChat MCP Integration Workflow

## Architecture Overview

```
Claude Code → MCP Tools → claude-cli-mcp server → AIChat HTTP API → LLM Response
```

## Available Access Methods

### 1. Direct MCP Tools (Current Session)
You can use the MCP tools that are already configured:

```javascript
// Through claude-cli MCP tools (when available in session)
mcp__claude-cli__claude_query("What is 2+2?")
mcp__claude-cli__claude_process_file("/path/to/file.py", "Analyze this code")
mcp__claude-cli__aichat_code_analysis_auto("def hello(): print('world')")
```

### 2. HTTP API Direct Access
Since AIChat server is running on 127.0.0.1:42333:

```bash
# Chat completion
curl -X POST http://127.0.0.1:42333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Explain Python decorators"}]
  }'

# Available models
curl http://127.0.0.1:42333/v1/models

# Embeddings
curl -X POST http://127.0.0.1:42333/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "input": "Text to embed"
  }'
```

### 3. Web Interface Access
Open your browser to:
- **Playground**: http://127.0.0.1:42333/playground
- **Arena**: http://127.0.0.1:42333/arena
- **API Docs**: http://127.0.0.1:42333/ (if available)

### 4. CLI Direct Usage
```bash
# Direct CLI usage (in current directory)
aichat "What is the weather today?"
aichat -f myfile.py "Explain this code"
aichat -r assistant "You are a helpful coding assistant"
```

## Configuration Files

### AIChat Config (~/.config/aichat/config.yaml)
```yaml
model: claude:claude-3-5-sonnet-20241022
temperature: 0.7
save: true
keybindings: emacs
```

### MCP Server Config (/home/bryan/mcp-servers-config.json)
```json
{
  "claude-cli": {
    "command": "/home/bryan/mcp-servers/claude-cli-mcp/venv/bin/python",
    "args": ["-m", "claude_cli_mcp.main"],
    "env": {
      "CLAUDE_CLI_COMMAND": "aichat",
      "AICHAT_SERVER_URL": "http://127.0.0.1:42333"
    }
  }
}
```

## Practical Examples

### Example 1: Code Analysis Through MCP
1. File analysis via claude-cli-mcp tools
2. Real-time feedback through HTTP API
3. Results stored in memory systems

### Example 2: Multi-Model Comparison
1. Use Arena interface (port 42333/arena)
2. Compare responses from different models
3. Analyze performance metrics

### Example 3: RAG Integration
1. Add documents via CLI: `aichat --rag myproject`
2. Query with context via API
3. Embed and search via endpoints

## Troubleshooting

### Server Not Responding
```bash
# Check if server is running
ps aux | grep aichat
curl http://127.0.0.1:42333/v1/models

# Restart if needed
pkill aichat
aichat --serve 127.0.0.1:42333 &
```

### MCP Integration Issues
```bash
# Check MCP server
ps aux | grep claude-cli-mcp

# Test MCP environment
/home/bryan/mcp-servers/claude-cli-mcp/venv/bin/python -c "import requests; print('✅ OK')"
```

### Configuration Problems
```bash
# Verify config files
cat ~/.config/aichat/config.yaml
cat /home/bryan/mcp-servers-config.json
```

## Advanced Features

### Function Calling
```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "What's the weather?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ]
}
```

### Session Management
```bash
# Save conversation
aichat -s weather_session "What's the weather?"

# Resume session
aichat -s weather_session "What about tomorrow?"
```

### Model Selection
```bash
# List available models
aichat --list-models

# Use specific model
aichat -m claude:claude-3-5-sonnet-20241022 "Hello"

# Via API
curl -X POST http://127.0.0.1:42333/v1/chat/completions \
  -d '{"model": "claude:claude-3-5-sonnet-20241022", ...}'
```