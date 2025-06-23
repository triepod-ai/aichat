# AIChat Troubleshooting Guide

## Common Issues and Solutions

### 1. "error decoding response body: expected value at line 1 column 1"

**Cause:** Authentication failure with your configured model provider.

**Solutions:**
```bash
# For VertexAI
gcloud auth application-default login
# or set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# For Claude
# Ensure your API key is valid in config.yaml

# For OpenAI
# Verify API key in config.yaml

# Test your model
aichat --model <your-model> "test message"
```

### 2. "max_tokens: Field required"

**Cause:** Some providers require explicit max_tokens configuration.

**Solution:**
Add to your model configuration:
```yaml
clients:
- type: claude
  api_key: your-key
  models:
    - name: claude-3-haiku
      max_tokens: 4096  # Add this line
```

### 3. RAG Database Works But Chat Fails

**Cause:** RAG database creation succeeded but chat model is misconfigured.

**Solution:**
```bash
# Use a working model explicitly
aichat --model ollama:llama3.2:1b --rag your-rag-name

# Or fix your default model in config.yaml
model: ollama:llama3.2:1b  # Change to working model
```

### 4. Function Calling Not Working

**Cause:** Model doesn't support function calling or functions not properly configured.

**Solution:**
```bash
# Ensure model supports function calling
aichat --model claude:claude-3-haiku  # Known to support functions

# Check function configuration
ls ~/.config/aichat/functions/
```

### 5. Server Won't Start on Port 8000

**Cause:** Port already in use or permission issues.

**Solution:**
```bash
# Use different port
aichat --serve --port 42333

# Or configure in config.yaml
serve_addr: 127.0.0.1:42333
```

## Model Configuration Testing

### Quick Model Test
```bash
# Test each configured model
aichat --model vertexai:gemini-2.0-flash-thinking-exp-01-21 "hello"
aichat --model claude:claude-3-haiku "hello"
aichat --model ollama:llama3.2:1b "hello"
```

### Find Working Model
```bash
# List available models
aichat --list-models

# Test with simple local model
aichat --model ollama:llama3.2:1b "test"
```

## RAG Troubleshooting

### Test RAG Database
```bash
# Verify RAG database exists
ls ~/.config/aichat/rags/

# Test with working model
aichat --model ollama:llama3.2:1b --rag your-rag-name "test query"
```

### Common RAG Issues
- **Empty responses:** Check if documents loaded properly
- **API errors:** Model configuration issue, not RAG issue
- **Slow responses:** Large database or complex model

## Configuration Validation

### Check Configuration
```bash
# View current config
aichat --info

# Validate specific settings
grep -A5 "model:" ~/.config/aichat/config.yaml
grep -A10 "clients:" ~/.config/aichat/config.yaml
```

### Minimal Working Configuration
```yaml
model: ollama:llama3.2:1b
function_calling: true

clients:
- type: openai-compatible
  name: ollama
  api_base: http://localhost:11434/v1
  api_key: dummy
  models:
    - name: llama3.2:1b
      max_input_tokens: 128000
      supports_function_calling: true
```

## Environment Issues

### WSL/Linux Specific
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check network connectivity
ping api.anthropic.com
ping api.openai.com
```

### Permission Issues
```bash
# Check config directory permissions
ls -la ~/.config/aichat/

# Fix permissions if needed
chmod 755 ~/.config/aichat/
chmod 644 ~/.config/aichat/config.yaml
```

## Getting Help

1. **Check logs:** AIChat provides detailed error messages
2. **Test incrementally:** Start with simple local models
3. **Verify authentication:** Each provider has different auth requirements
4. **Use fallbacks:** Keep a working local model (Ollama) as backup

## Success Indicators

✅ **Working setup:**
```bash
$ aichat "hello"
Hello! How can I help you today?

$ aichat --serve
Chat Completions API: http://127.0.0.1:42333/v1/chat/completions
...

$ aichat --rag test-rag "what is this about?"
[Relevant response from RAG database]
```

## Emergency Fallback

If nothing works, use this minimal config:
```yaml
model: ollama:llama3.2:1b

clients:
- type: openai-compatible
  name: ollama
  api_base: http://localhost:11434/v1
```

Then run: `ollama pull llama3.2:1b`