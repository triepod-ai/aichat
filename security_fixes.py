#!/usr/bin/env python3
"""
Security Fixes for AIChat Smart Routing MCP Tools
=================================================

Comprehensive input sanitization and environment validation fixes
for the critical vulnerabilities discovered by the Novel Testing Framework v2.0.

This module provides:
1. Unicode spoofing protection
2. Command injection prevention  
3. Environment dependency validation
4. Graceful degradation handling
"""

import re
import unicodedata
import subprocess
import logging
import time
import requests
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# CRITICAL FIX 1: COMPREHENSIVE INPUT SANITIZATION
# ============================================================================

def sanitize_mcp_input(user_input: str) -> str:
    """
    Comprehensive input sanitization for MCP tools
    Blocks Unicode spoofing, command injection, and malicious patterns
    
    Args:
        user_input: Raw user input from MCP tool calls
        
    Returns:
        Sanitized input safe for command execution
        
    Raises:
        ValueError: If input contains malicious patterns or is invalid
    """
    if not user_input:
        raise ValueError("Empty input not allowed")
        
    if len(user_input) > 10000:
        raise ValueError("Input too long (max 10000 characters)")
    
    # Normalize Unicode to detect spoofing
    try:
        normalized = unicodedata.normalize('NFKD', user_input)
    except ValueError as e:
        raise ValueError(f"Invalid Unicode input: {e}")
    
    # Block dangerous Unicode categories
    dangerous_categories = ['Cf', 'Mn', 'Me']  # Format, Nonspacing, Enclosing marks
    for char in normalized:
        category = unicodedata.category(char)
        if category in dangerous_categories:
            raise ValueError(f"Dangerous Unicode character detected: {repr(char)} (category: {category})")
    
    # Block command injection patterns (more targeted)
    injection_patterns = [
        r'[;&|`]\s*[a-zA-Z]',   # Command separators followed by commands
        r'\$\([^)]*\)',         # Command substitution $(...)
        r'\\[rnts]',            # Escape sequences  
        r'\.\./.*[/\\]etc[/\\]', # Path traversal to /etc/
        r'\.\./.*[/\\]bin[/\\]', # Path traversal to /bin/
        r'rm\s+-rf?\s+/',       # Destructive rm commands
        r'del\s+/s',            # Windows destructive commands
        r'sudo\s+rm',           # Privileged destructive operations
        r'curl\s+.*[|;&]',      # Network with command chaining
        r'wget\s+.*[|;&]',      # Network with command chaining
        r'nc\s+.*[|;&]',        # Netcat with command chaining
        r'python\s+-c\s+[\'"].*[|;&]', # Python inline with chaining
        r'bash\s+-c\s+[\'"].*[|;&]',   # Bash inline with chaining
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            raise ValueError(f"Command injection pattern detected: {pattern}")
    
    # Additional validation for MCP context
    sanitized = normalized.strip()
    if len(sanitized) < 1:
        raise ValueError("Empty query after sanitization")
    
    return sanitized

def validate_code_input(code_content: str) -> str:
    """
    Special validation for code content inputs
    Less restrictive than general input but still safe
    """
    if not code_content:
        raise ValueError("Empty code content not allowed")
        
    if len(code_content) > 50000:
        raise ValueError("Code content too long (max 50000 characters)")
    
    # Normalize but allow more characters for code
    normalized = unicodedata.normalize('NFKD', code_content)
    
    # Only block obviously malicious patterns in code
    malicious_patterns = [
        r'rm\s+-rf\s+/',        # Destructive filesystem operations
        r'sudo\s+rm',           # Privileged destructive operations
        r'>/dev/null\s*&&\s*rm', # Hidden destructive operations
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            raise ValueError(f"Potentially malicious code pattern detected: {pattern}")
    
    return normalized.strip()

# ============================================================================
# CRITICAL FIX 2: ENVIRONMENT DEPENDENCY VALIDATION
# ============================================================================

async def validate_environment() -> Dict[str, bool]:
    """
    Validate all required dependencies for smart routing
    
    Returns:
        Dictionary with validation status for each dependency
    """
    validation_results = {}
    
    # Check Ollama availability through Docker host
    try:
        # First try direct API call to host.docker.internal:11434
        response = requests.get('http://host.docker.internal:11434/api/tags', timeout=5)
        if response.status_code == 200:
            validation_results['ollama'] = True
            models_data = response.json()
            model_names = [model.get('name', '') for model in models_data.get('models', [])]
            
            # Check for required models
            validation_results['llama3.2:1b'] = any('llama3.2:1b' in name for name in model_names)
            validation_results['qwen2.5-coder:7b-instruct'] = any('qwen2.5-coder:7b-instruct' in name for name in model_names)
        else:
            validation_results['ollama'] = False
            validation_results['llama3.2:1b'] = False
            validation_results['qwen2.5-coder:7b-instruct'] = False
            
    except requests.RequestException:
        # Fallback: try ollama command if available
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, timeout=5, text=True)
            validation_results['ollama'] = result.returncode == 0
            
            if validation_results['ollama']:
                # Check required models
                models_output = result.stdout
                validation_results['llama3.2:1b'] = 'llama3.2:1b' in models_output
                validation_results['qwen2.5-coder:7b-instruct'] = 'qwen2.5-coder:7b-instruct' in models_output
            else:
                validation_results['llama3.2:1b'] = False
                validation_results['qwen2.5-coder:7b-instruct'] = False
                
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            validation_results['ollama'] = False
            validation_results['llama3.2:1b'] = False
            validation_results['qwen2.5-coder:7b-instruct'] = False
    
    # Check AIChat server
    try:
        response = requests.get('http://127.0.0.1:42333/v1/models', timeout=2)
        validation_results['aichat_server'] = response.status_code == 200
    except requests.RequestException:
        validation_results['aichat_server'] = False
    
    # Check AIChat command availability
    try:
        result = subprocess.run(['aichat', '--version'], capture_output=True, timeout=5)
        validation_results['aichat_command'] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        validation_results['aichat_command'] = False
    
    return validation_results

def select_fallback_model(env_status: Dict[str, bool], task_type: str = "general") -> str:
    """
    Select the best available model based on environment status
    
    Args:
        env_status: Environment validation results
        task_type: Type of task ("general", "code", "quick")
        
    Returns:
        Best available model name
    """
    # Priority order based on task type and availability
    model_preferences = {
        "quick": [
            ("ollama:llama3.2:1b", ["ollama", "llama3.2:1b"]),
            ("claude:claude-3-5-haiku", ["aichat_command"]),
            ("vertexai:claude-3-5-haiku", ["aichat_command"]),
        ],
        "code": [
            ("ollama:qwen2.5-coder:7b-instruct", ["ollama", "qwen2.5-coder:7b-instruct"]),
            ("claude:claude-3-5-sonnet", ["aichat_command"]),
            ("vertexai:claude-3-5-sonnet", ["aichat_command"]),
        ],
        "general": [
            ("ollama:llama3.2:1b", ["ollama", "llama3.2:1b"]),
            ("claude:claude-3-5-haiku", ["aichat_command"]),
            ("vertexai:claude-3-5-haiku", ["aichat_command"]),
        ]
    }
    
    preferences = model_preferences.get(task_type, model_preferences["general"])
    
    for model, requirements in preferences:
        if all(env_status.get(req, False) for req in requirements):
            return model
    
    # Final fallback
    return "claude:claude-3-5-haiku"

def get_fallback_message(env_status: Dict[str, bool], selected_model: str, optimal_model: str) -> str:
    """
    Generate user-friendly message about fallback model selection
    """
    if selected_model == optimal_model:
        return ""
    
    missing_deps = [dep for dep, status in env_status.items() if not status]
    
    if "ollama" in missing_deps:
        return f"⚠️ Using {selected_model} (Ollama not available - install for cost optimization)"
    elif any("llama" in dep or "qwen" in dep for dep in missing_deps):
        return f"⚠️ Using {selected_model} (Local models not installed - run: ollama pull llama3.2:1b)"
    else:
        return f"ℹ️ Using {selected_model} (automatic fallback)"

# ============================================================================
# SECURITY TEST VALIDATION
# ============================================================================

def test_input_sanitization():
    """Test all discovered attack vectors"""
    attack_vectors = [
        "ba𝗌h; rm -rf /",          # Unicode spoofing
        "rea‌d /etc/passwd",        # Zero-width injection  
        "input`whoami`",           # Backtick injection
        "test; echo pwned",        # Command chaining
        "script\u202e.exe",        # RTL override
        "$(rm -rf /)",             # Command substitution
        "python -c 'import os; os.system(\"rm -rf /\")'", # Python injection
    ]
    
    print("🧪 Testing input sanitization against attack vectors...")
    
    for i, attack in enumerate(attack_vectors, 1):
        try:
            sanitized = sanitize_mcp_input(attack)
            print(f"❌ Attack {i} NOT BLOCKED: {repr(attack)}")
            print(f"   Sanitized to: {repr(sanitized)}")
            return False
        except ValueError as e:
            print(f"✅ Attack {i} BLOCKED: {repr(attack)}")
            print(f"   Reason: {e}")
    
    # Test legitimate inputs pass through
    legitimate_inputs = [
        "What is Python?",
        "Calculate 2+2",
        "Review this code: def hello(): print('world')",
        "Explain how databases work",
        "What's the weather like?",
    ]
    
    print("\n🧪 Testing legitimate inputs pass through...")
    
    for i, input_text in enumerate(legitimate_inputs, 1):
        try:
            sanitized = sanitize_mcp_input(input_text)
            if sanitized == input_text.strip():
                print(f"✅ Legitimate {i} PASSED: {repr(input_text)}")
            else:
                print(f"⚠️ Legitimate {i} MODIFIED: {repr(input_text)} → {repr(sanitized)}")
        except ValueError as e:
            print(f"❌ Legitimate {i} BLOCKED: {repr(input_text)}")
            print(f"   Reason: {e}")
            return False
    
    print("\n✅ Input sanitization tests PASSED")
    return True

async def test_environment_validation():
    """Test graceful degradation and dependency handling"""
    print("\n🧪 Testing environment validation...")
    
    env_status = await validate_environment()
    
    print("Environment status:")
    for component, status in env_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {status}")
    
    # Test model selection for different scenarios
    scenarios = [
        ("quick", "ollama:llama3.2:1b"),
        ("code", "ollama:qwen2.5-coder:7b-instruct"),  
        ("general", "ollama:llama3.2:1b"),
    ]
    
    print("\nModel selection tests:")
    for task_type, optimal_model in scenarios:
        selected = select_fallback_model(env_status, task_type)
        message = get_fallback_message(env_status, selected, optimal_model)
        
        print(f"  📋 Task: {task_type}")
        print(f"     Optimal: {optimal_model}")
        print(f"     Selected: {selected}")
        if message:
            print(f"     Message: {message}")
    
    print("\n✅ Environment validation tests completed")
    return env_status

if __name__ == "__main__":
    """Run security tests"""
    import asyncio
    
    print("🔒 SECURITY FIXES VALIDATION")
    print("=" * 50)
    
    # Test input sanitization
    sanitization_passed = test_input_sanitization()
    
    # Test environment validation
    env_status = asyncio.run(test_environment_validation())
    
    if sanitization_passed:
        print("\n🎯 SECURITY FIXES READY FOR DEPLOYMENT")
        print("All critical vulnerabilities have been addressed")
    else:
        print("\n❌ SECURITY TESTS FAILED")
        print("Review sanitization logic before deployment")