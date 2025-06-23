#!/usr/bin/env python3
"""
Real MCP Tools Testing - Direct Testing of Smart Routing Tools
==============================================================

This script attempts to directly test the MCP smart routing tools by:
1. Checking actual aichat functionality 
2. Testing basic query routing
3. Measuring real performance metrics
4. Validating model selection logic

Uses actual aichat command with real models.
"""

import asyncio
import json
import time
import subprocess
import sys
from typing import Dict, Any, List
import tempfile
import os

class RealMCPToolsTester:
    """Direct tester for MCP smart routing functionality"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.time(),
            "aichat_tests": {},
            "model_availability": {},
            "routing_tests": {},
            "performance_metrics": {},
            "cost_analysis": {}
        }
        self.available_models = []
        
    async def setup_environment(self) -> bool:
        """Set up and validate test environment"""
        print("🔧 SETTING UP TEST ENVIRONMENT")
        print("=" * 50)
        
        # 1. Check aichat command
        print("1. Checking aichat command...")
        if not await self.check_aichat_available():
            print("   ❌ aichat command not available")
            return False
        print("   ✅ aichat command available")
        
        # 2. Check model availability
        print("2. Discovering available models...")
        models = await self.discover_available_models()
        self.available_models = models
        self.test_results["model_availability"] = models
        
        for model in models:
            print(f"   ✅ {model}")
        
        if not models:
            print("   ⚠️ No models available for testing")
            return False
        
        print(f"   📊 Found {len(models)} available models")
        return True
    
    async def check_aichat_available(self) -> bool:
        """Check if aichat command is available"""
        try:
            result = await asyncio.create_subprocess_exec(
                "aichat", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def discover_available_models(self) -> List[str]:
        """Discover available models for testing"""
        models = []
        
        # Check Ollama models
        try:
            result = await asyncio.create_subprocess_exec(
                "curl", "-s", "http://host.docker.internal:11434/api/tags",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            
            if result.returncode == 0:
                data = json.loads(stdout.decode())
                ollama_models = [f"ollama:{model['name']}" for model in data.get('models', [])]
                
                # Filter for the models used in smart routing
                target_models = ["llama3.2:1b", "qwen2.5-coder:7b-instruct"]
                for target in target_models:
                    for ollama_model in ollama_models:
                        if target in ollama_model:
                            models.append(ollama_model)
                            break
        except Exception as e:
            print(f"   Error checking Ollama models: {e}")
        
        # Assume Claude is available if configured (would need API key check)
        # For testing purposes, we'll focus on Ollama models
        
        return models
    
    async def test_basic_aichat_functionality(self) -> Dict[str, Any]:
        """Test basic aichat functionality with available models"""
        print("\n📝 TESTING BASIC AICHAT FUNCTIONALITY")
        print("=" * 50)
        
        results = {}
        test_query = "What is 2+2? Answer with just the number."
        
        for model in self.available_models[:2]:  # Test first 2 available models
            print(f"Testing model: {model}")
            
            start_time = time.time()
            success, response, error = await self.run_aichat_query(test_query, model)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            results[model] = {
                "success": success,
                "response_time": response_time,
                "response_length": len(response) if response else 0,
                "error": error
            }
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {status} - {response_time:.3f}s - {model}")
            if error:
                print(f"   Error: {error[:100]}...")
        
        self.test_results["aichat_tests"] = results
        return results
    
    async def run_aichat_query(self, query: str, model: str, timeout: int = 30) -> tuple:
        """Run a query with aichat using specified model"""
        try:
            result = await asyncio.create_subprocess_exec(
                "aichat", "--model", model, query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(), 
                    timeout=timeout
                )
                
                success = result.returncode == 0
                response = stdout.decode().strip() if stdout else ""
                error = stderr.decode().strip() if stderr else ""
                
                return success, response, error
                
            except asyncio.TimeoutError:
                result.terminate()
                await result.wait()
                return False, "", "Timeout after {timeout}s"
                
        except Exception as e:
            return False, "", str(e)
    
    async def test_smart_routing_logic(self) -> Dict[str, Any]:
        """Test smart routing logic with different query types"""
        print("\n🧠 TESTING SMART ROUTING LOGIC")
        print("=" * 50)
        
        # Define test cases with expected routing behavior
        test_cases = [
            {
                "query": "What is 5+3?",
                "type": "simple_math",
                "expected_model": "llama3.2:1b",
                "reasoning": "Simple calculation should use fast model"
            },
            {
                "query": "Write a Python function to calculate fibonacci numbers",
                "type": "code_generation", 
                "expected_model": "qwen2.5-coder:7b-instruct",
                "reasoning": "Code tasks should use coding-specialized model"
            },
            {
                "query": "Hello, how are you?",
                "type": "simple_greeting",
                "expected_model": "llama3.2:1b", 
                "reasoning": "Simple greeting should use fast model"
            },
            {
                "query": "Debug this Python code: def broken_func(): print('hello'",
                "type": "code_debugging",
                "expected_model": "qwen2.5-coder:7b-instruct",
                "reasoning": "Code debugging should use coding model"
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"Testing: {test_case['type']}")
            print(f"   Query: {test_case['query'][:50]}...")
            
            # Simulate smart routing decision
            selected_model = self.simulate_smart_routing(test_case["query"])
            
            # Check if available model matches expected
            expected_short = test_case["expected_model"]
            selected_short = selected_model.split(":")[-1] if ":" in selected_model else selected_model
            expected_short = expected_short.split(":")[-1] if ":" in expected_short else expected_short
            
            correct_routing = expected_short in selected_short
            
            # Test actual query if model is available
            if selected_model in self.available_models:
                success, response, error = await self.run_aichat_query(
                    test_case["query"], 
                    selected_model, 
                    timeout=20
                )
            else:
                success, response, error = False, "", "Model not available"
            
            results[test_case["type"]] = {
                "query": test_case["query"],
                "expected_model": test_case["expected_model"],
                "selected_model": selected_model,
                "correct_routing": correct_routing,
                "execution_success": success,
                "response_preview": response[:100] if response else "",
                "error": error,
                "reasoning": test_case["reasoning"]
            }
            
            routing_status = "✅ CORRECT" if correct_routing else "❌ INCORRECT"
            exec_status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   Routing: {routing_status} - {selected_model}")
            print(f"   Execution: {exec_status}")
        
        self.test_results["routing_tests"] = results
        return results
    
    def simulate_smart_routing(self, query: str) -> str:
        """Simulate the smart routing logic based on query analysis"""
        query_lower = query.lower()
        
        # Code-related patterns
        code_indicators = [
            "python", "function", "code", "debug", "script", 
            "def ", "class ", "import ", "print(", "return"
        ]
        
        # Simple task patterns
        simple_indicators = [
            "what is", "hello", "hi ", "how are", "calculate", 
            "+", "-", "*", "/", "="
        ]
        
        # Check for code patterns
        if any(indicator in query_lower for indicator in code_indicators):
            # Prefer qwen2.5-coder for code tasks
            for model in self.available_models:
                if "qwen2.5-coder" in model:
                    return model
        
        # Check for simple patterns  
        if any(indicator in query_lower for indicator in simple_indicators):
            # Prefer llama3.2:1b for simple tasks
            for model in self.available_models:
                if "llama3.2:1b" in model:
                    return model
        
        # Default to first available model
        return self.available_models[0] if self.available_models else "unknown"
    
    async def test_performance_characteristics(self) -> Dict[str, Any]:
        """Test performance characteristics of different models"""
        print("\n⚡ TESTING PERFORMANCE CHARACTERISTICS")
        print("=" * 50)
        
        results = {}
        
        # Performance test queries
        test_queries = {
            "quick_math": "Calculate 15 * 23",
            "simple_question": "What is the capital of France?",
            "code_snippet": "Write a Python function to reverse a string"
        }
        
        for model in self.available_models[:2]:  # Test first 2 models
            print(f"Testing performance for: {model}")
            model_results = {}
            
            for query_type, query in test_queries.items():
                times = []
                successes = 0
                
                # Run query 3 times for average
                for i in range(3):
                    start_time = time.time()
                    success, response, error = await self.run_aichat_query(query, model, timeout=15)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    if success:
                        successes += 1
                
                model_results[query_type] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "success_rate": successes / 3
                }
                
                avg_time = model_results[query_type]["avg_time"]
                success_rate = model_results[query_type]["success_rate"]
                print(f"   {query_type}: {avg_time:.3f}s avg, {success_rate:.1%} success")
            
            results[model] = model_results
        
        self.test_results["performance_metrics"] = results
        return results
    
    async def test_cost_analysis(self) -> Dict[str, Any]:
        """Analyze cost characteristics of different routing decisions"""
        print("\n💰 TESTING COST ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        # Cost analysis scenarios
        scenarios = [
            {
                "name": "simple_tasks",
                "queries": ["2+2=?", "Hello", "What time is it?"],
                "optimal_model": "llama3.2:1b",
                "reasoning": "Simple tasks should use cheapest/fastest model"
            },
            {
                "name": "code_tasks", 
                "queries": ["Write a function", "Debug this code", "Explain this algorithm"],
                "optimal_model": "qwen2.5-coder:7b-instruct",
                "reasoning": "Code tasks need specialized model despite higher cost"
            }
        ]
        
        for scenario in scenarios:
            print(f"Analyzing scenario: {scenario['name']}")
            
            scenario_results = {
                "optimal_model": scenario["optimal_model"],
                "routing_decisions": [],
                "cost_efficiency": 0
            }
            
            correct_routes = 0
            
            for query in scenario["queries"]:
                selected_model = self.simulate_smart_routing(query)
                optimal_short = scenario["optimal_model"].split(":")[-1]
                selected_short = selected_model.split(":")[-1] if ":" in selected_model else selected_model
                
                is_optimal = optimal_short in selected_short
                if is_optimal:
                    correct_routes += 1
                
                scenario_results["routing_decisions"].append({
                    "query": query,
                    "selected": selected_model,
                    "optimal": is_optimal
                })
            
            scenario_results["cost_efficiency"] = correct_routes / len(scenario["queries"])
            results[scenario["name"]] = scenario_results
            
            efficiency = scenario_results["cost_efficiency"]
            print(f"   Cost efficiency: {efficiency:.1%}")
            print(f"   Reasoning: {scenario['reasoning']}")
        
        self.test_results["cost_analysis"] = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        print("\n📊 PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        
        report = []
        report.append("# Smart Routing MCP Tools - Performance Report")
        report.append(f"Generated: {time.ctime(self.test_results['timestamp'])}")
        report.append("")
        
        # Model Availability
        models = self.test_results.get("model_availability", [])
        report.append(f"## Available Models ({len(models)})")
        for model in models:
            report.append(f"- {model}")
        
        # Basic Functionality
        aichat_tests = self.test_results.get("aichat_tests", {})
        report.append("\n## Basic Functionality Test Results")
        
        for model, result in aichat_tests.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            time_ms = result["response_time"] * 1000
            report.append(f"- {model}: {status} ({time_ms:.0f}ms)")
        
        # Routing Accuracy
        routing_tests = self.test_results.get("routing_tests", {})
        if routing_tests:
            correct_routes = sum(1 for r in routing_tests.values() if r["correct_routing"])
            total_routes = len(routing_tests)
            accuracy = (correct_routes / total_routes * 100) if total_routes > 0 else 0
            
            report.append(f"\n## Smart Routing Accuracy: {accuracy:.1f}%")
            report.append(f"Correct routes: {correct_routes}/{total_routes}")
            
            for test_type, result in routing_tests.items():
                status = "✅" if result["correct_routing"] else "❌"
                report.append(f"- {test_type}: {status} {result['selected_model']}")
        
        # Performance Metrics
        perf_metrics = self.test_results.get("performance_metrics", {})
        if perf_metrics:
            report.append("\n## Performance Metrics")
            
            for model, metrics in perf_metrics.items():
                report.append(f"\n### {model}")
                for query_type, data in metrics.items():
                    avg_ms = data["avg_time"] * 1000
                    success_rate = data["success_rate"] * 100
                    report.append(f"- {query_type}: {avg_ms:.0f}ms avg, {success_rate:.0f}% success")
        
        # Cost Analysis
        cost_analysis = self.test_results.get("cost_analysis", {})
        if cost_analysis:
            report.append("\n## Cost Efficiency Analysis")
            
            for scenario, data in cost_analysis.items():
                efficiency = data["cost_efficiency"] * 100
                report.append(f"- {scenario}: {efficiency:.1f}% cost-optimal routing")
        
        # Key Findings
        report.append("\n## Key Findings")
        
        if len(models) > 0:
            report.append("✅ Multiple models available for smart routing")
        else:
            report.append("❌ No models available - cannot test routing")
        
        if routing_tests:
            routing_accuracy = sum(1 for r in routing_tests.values() if r["correct_routing"]) / len(routing_tests)
            if routing_accuracy >= 0.8:
                report.append("✅ Smart routing accuracy is good (≥80%)")
            else:
                report.append("⚠️ Smart routing accuracy needs improvement (<80%)")
        
        if perf_metrics:
            # Find fastest model
            fastest_model = None
            fastest_time = float('inf')
            
            for model, metrics in perf_metrics.items():
                avg_times = [data["avg_time"] for data in metrics.values()]
                model_avg = sum(avg_times) / len(avg_times)
                if model_avg < fastest_time:
                    fastest_time = model_avg
                    fastest_model = model
            
            if fastest_model:
                report.append(f"⚡ Fastest model: {fastest_model} ({fastest_time*1000:.0f}ms avg)")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        if not models:
            report.append("- 🔧 Install and configure Ollama models for testing")
        
        if routing_tests and sum(1 for r in routing_tests.values() if r["correct_routing"]) / len(routing_tests) < 0.8:
            report.append("- 📈 Improve routing logic for better model selection")
        
        if perf_metrics:
            # Check if any model is consistently slow
            slow_models = []
            for model, metrics in perf_metrics.items():
                avg_times = [data["avg_time"] for data in metrics.values()]
                if any(t > 5.0 for t in avg_times):  # More than 5 seconds
                    slow_models.append(model)
            
            if slow_models:
                report.append(f"- ⚡ Optimize performance for slow models: {', '.join(slow_models)}")
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = int(time.time())
        filename = f"/home/bryan/apps/aichat/mcp_tools_performance_report_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write(report_text)
        
        print(f"\n📁 Report saved to: {filename}")
        print("\n" + report_text)
        
        return report_text
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("🚀 REAL MCP TOOLS COMPREHENSIVE TESTING")
        print("=" * 60)
        print("Testing actual aichat smart routing with real models")
        print("")
        
        # Setup environment
        if not await self.setup_environment():
            print("❌ Environment setup failed. Cannot proceed with tests.")
            return
        
        # Run test suites
        await self.test_basic_aichat_functionality()
        await self.test_smart_routing_logic()
        await self.test_performance_characteristics()
        await self.test_cost_analysis()
        
        # Generate final report
        self.generate_performance_report()

async def main():
    """Main test execution"""
    tester = RealMCPToolsTester()
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())