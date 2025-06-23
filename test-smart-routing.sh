#!/bin/bash

# Test script for Smart Routing MCP Tools
echo "🧪 Testing Smart Routing MCP Tools"
echo "=================================="

# Wait for MCP server to be ready
echo "⏳ Waiting for MCP server to initialize..."
sleep 5

echo "📊 Testing available models..."
aichat --list-models | head -10

echo "🚀 Testing AIChat server status..."
curl -s http://127.0.0.1:42333/v1/models | jq '.data[0:3]' 2>/dev/null || echo "Server not responding"

echo "📚 Testing RAG database..."
aichat --rag repo-knowledge "What is AIChat?" | head -5

echo "🔧 Testing Ollama models..."
echo "Available Ollama models:"
ollama list | grep -E "(llama3.2:1b|qwen2.5-coder)"

echo "✅ Setup verification complete!"
echo ""
echo "💡 Next steps:"
echo "1. Test MCP tools via Claude Code:"
echo "   mcp__claude-cli__aichat_quick_task('What is 2+2?')"
echo "2. Test smart routing:"
echo "   mcp__claude-cli__aichat_smart_route('Simple question', '', 'speed')"
echo "3. Test code analysis:"
echo "   mcp__claude-cli__aichat_code_task('def hello(): print(\"world\")', 'review')"
echo "4. Test RAG queries:"
echo "   mcp__claude-cli__aichat_rag_query('How to use AIChat?')"