# Smart Routing MCP Tools - Performance Report
Generated: Sun Jun 22 20:59:22 2025

## Available Models (2)
- ollama:llama3.2:1b
- ollama:qwen2.5-coder:7b-instruct

## Basic Functionality Test Results
- ollama:llama3.2:1b: ✅ PASS (1570ms)
- ollama:qwen2.5-coder:7b-instruct: ✅ PASS (3309ms)

## Smart Routing Accuracy: 100.0%
Correct routes: 4/4
- simple_math: ✅ ollama:llama3.2:1b
- code_generation: ✅ ollama:qwen2.5-coder:7b-instruct
- simple_greeting: ✅ ollama:llama3.2:1b
- code_debugging: ✅ ollama:qwen2.5-coder:7b-instruct

## Performance Metrics

### ollama:llama3.2:1b
- quick_math: 254ms avg, 100% success
- simple_question: 69ms avg, 100% success
- code_snippet: 1087ms avg, 100% success

### ollama:qwen2.5-coder:7b-instruct
- quick_math: 336ms avg, 100% success
- simple_question: 556ms avg, 100% success
- code_snippet: 1356ms avg, 100% success

## Cost Efficiency Analysis
- simple_tasks: 100.0% cost-optimal routing
- code_tasks: 66.7% cost-optimal routing

## Key Findings
✅ Multiple models available for smart routing
✅ Smart routing accuracy is good (≥80%)
⚡ Fastest model: ollama:llama3.2:1b (470ms avg)

## Recommendations