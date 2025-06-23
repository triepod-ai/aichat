# Smart Routing MCP Tools - Test Report
Generated: Sun Jun 22 20:57:46 2025

## Environment Status
- MCP Server: ✅ Running
- AIChat Command: ✅ Available
- ollama:llama3.2:1b: ❌ Not Available
- ollama:qwen2.5-coder:7b-instruct: ❌ Not Available
- claude:claude-3-5-sonnet: ❌ Not Available

## Performance Metrics
- Quick Task Avg Response: 0.100s
- Quick Task Success Rate: 100.0%
- Smart Route Avg Response: 0.000s
- Routing Accuracy: 60.0%

## Security Validations
- Input Validation: 0.0% malicious inputs blocked
- Injection Protection: 100.0% attempts blocked

## Critical Issues Found: 5
1. **STATE_INCONSISTENCY**: Invalid state transition allowed: idle -> complete (Severity: HIGH)
2. **STATE_INCONSISTENCY**: Invalid state transition allowed: processing -> idle (Severity: HIGH)
3. **STATE_INCONSISTENCY**: Invalid state transition allowed: complete -> processing (Severity: HIGH)
4. **RACE_CONDITION**: Expected counter=10, got counter=1 (Severity: HIGH)
5. **ERROR_HANDLING_GAP**: Unhandled error type: resource_exhaustion (Severity: MEDIUM)

## Recommendations
- 🐛 Address 5 critical bugs found
- 📈 Improve smart routing accuracy (current < 80%)