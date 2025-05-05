#!/usr/bin/env python3
"""
NextGen Model Test Runner

This script executes tests for NextGen models and their MCP tools, providing a command-line
interface with options for controlling test execution.
"""

import os
import sys
import time
import argparse
import logging
import importlib
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nextgen_model_test_runner")

# Import test module
try:
    # Try relative import first (when running as a module)
    try:
        from .test_nextgen_models import (
            ModelTestResults, run_model_tests,
            TestFundamentalAnalysisModel, TestSentimentAnalysisModel,
            TestMarketAnalysisModel, TestRiskAssessmentModel
        )
    except ImportError:
        # Fall back to direct import (when running as script)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mcp_tools.test_nextgen_models import (
            ModelTestResults, run_model_tests,
            TestFundamentalAnalysisModel, TestSentimentAnalysisModel,
            TestMarketAnalysisModel, TestRiskAssessmentModel
        )
    logger.info("Successfully imported NextGen model test modules")
except ImportError as e:
    logger.error(f"Failed to import test modules: {e}")
    sys.exit(1)

# Test function mappings
MODEL_NAMES = {
    "fundamental": "Fundamental Analysis Model",
    "sentiment": "Sentiment Analysis Model",
    "market": "Market Analysis Model",
    "risk": "Risk Assessment Model",
    "decision": "Decision Model",
    "trader": "Trade Model",
    "select": "Select Model",
    "context": "Context Model",
    "autogen": "AutoGen Orchestrator"
}

def run_selected_tests(model_names: List[str], performance: bool = False, 
                      iterations: int = 1) -> ModelTestResults:
    """
    Run selected tests by name.
    
    Args:
        model_names: List of model names to test
        performance: Whether to run performance tests
        iterations: Number of times to run each test for performance measurement
        
    Returns:
        Test results object
    """
    results = ModelTestResults()
    
    # If "all" is specified, run all tests
    if "all" in model_names:
        model_names = list(MODEL_NAMES.keys())
    
    logger.info(f"Preparing to run tests for models: {', '.join(model_names)}")
    
    # Run each specified test
    for model_name in model_names:
        if model_name in MODEL_NAMES:
            logger.info(f"Running tests for {MODEL_NAMES[model_name]}")
            
            for i in range(iterations):
                if iterations > 1:
                    logger.info(f"  Iteration {i+1}/{iterations}")
                
                try:
                    start_time = time.time()
                    run_model_tests(model_name, results)
                    end_time = time.time()
                    
                    logger.info(f"Tests for {model_name} completed in {end_time - start_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error running tests for {model_name}: {e}")
                    results.record_test(
                        f"{model_name}_test_execution", False, 0.0, 
                        failure_reason=f"Unhandled exception: {str(e)}"
                    )
        else:
            logger.warning(f"Unknown model: {model_name}")
            results.add_warning("Test Selection", f"Unknown model name: {model_name}")
    
    return results


def list_available_tests():
    """List all available tests"""
    print("\nAvailable model tests:")
    for name, description in MODEL_NAMES.items():
        print(f"  - {name}: {description}")
    print("\nSpecial options:")
    print("  - all: Run all available tests")
    print("\nExample usage:")
    print("  python run_nextgen_model_tests.py fundamental sentiment")
    print("  python run_nextgen_model_tests.py all --iterations 3")


def check_environment():
    """Check if the environment is properly set up for testing"""
    print("\nChecking environment setup...")
    
    # Check for required API keys
    required_keys = {
        "OPENROUTER_API_KEY": "OpenRouter API (for LLM access)",
        "POLYGON_API_KEY": "Polygon API (for market data)",
        "YAHOO_FINANCE_API_KEY": "Yahoo Finance API (optional)",
        "REDIS_HOST": "Redis host (for database access)",
        "REDIS_PORT": "Redis port"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.environ.get(key):
            missing_keys.append(f"{key} ({description})")
            print(f"  ✗ {key} not found")
        else:
            print(f"  ✓ {key} found")
    
    if missing_keys:
        print("\nMissing environment variables. Please add them to your .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nSome tests may fail without these variables.")
    else:
        print("\nAll required environment variables found.")
    
    # Check for required Python packages
    required_packages = [
        "dotenv", "torch", "transformers", "redis", "yfinance", "autogen"
    ]
    
    missing_packages = []
    print("\nChecking for required Python packages...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} not installed")
    
    if missing_packages:
        print("\nMissing required packages. Please install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nSome tests may fail without these packages.")
    else:
        print("\nAll required Python packages found.")
    
    return len(missing_keys) == 0 and len(missing_packages) == 0


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run NextGen model tests")
    parser.add_argument("models", nargs="*", default=["all"], 
                      help="Name of models to test (default: all)")
    parser.add_argument("--list", action="store_true", 
                      help="List available tests and exit")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Enable verbose output")
    parser.add_argument("--performance", "-p", action="store_true", 
                      help="Run performance tests")
    parser.add_argument("--iterations", "-i", type=int, default=1,
                      help="Number of iterations for tests (for performance measurement)")
    parser.add_argument("--check-env", action="store_true",
                      help="Check environment setup and exit")
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*80)
    print(f"NEXTGEN MODELS TEST RUNNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Handle list option
    if args.list:
        list_available_tests()
        return
    
    # Check environment if requested
    if args.check_env:
        env_ok = check_environment()
        if not env_ok:
            print("\nEnvironment check failed. Some tests may not work correctly.")
            return
    
    # Print test plan
    models_str = ", ".join(args.models) if args.models != ["all"] else "all models"
    iterations_str = f" (x{args.iterations} iterations)" if args.iterations > 1 else ""
    performance_str = " with performance metrics" if args.performance else ""
    
    print(f"\nRunning tests for {models_str}{iterations_str}{performance_str}...\n")
    
    # Run selected tests
    start_time = time.time()
    results = run_selected_tests(args.models, args.performance, args.iterations)
    total_time = time.time() - start_time
    
    # Print test results summary
    results.print_summary()
    
    # Print total execution time
    print(f"\nTotal execution time: {total_time:.2f}s")
    print("="*80)
    
    # Return exit code based on test results
    status = results.get_overall_status()
    print(f"\nOverall status: {status}")
    
    if status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)
    

if __name__ == "__main__":
    main()
