#!/usr/bin/env python3
"""
MCP Tools Test Runner

This script executes tests for MCP tools, providing a command-line interface
with options for controlling test execution.
"""

import os
import sys
import time
import argparse
import logging
import importlib
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_test_runner")

# Import test module
try:
    # Try relative import first (when running as a module)
    try:
        from .test_mcp_tools import (
            MCPTestResults, test_mcp_base, test_yahoo_finance_mcp,
            test_trading_mcp, test_financial_text_mcp, test_redis_mcp,
            test_polygon_rest_mcp, test_other_mcp_initialization,
            run_performance_test
        )
    except ImportError:
        # Fall back to direct import (when running as script)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mcp_tools.test_mcp_tools import (
            MCPTestResults, test_mcp_base, test_yahoo_finance_mcp,
            test_trading_mcp, test_financial_text_mcp, test_redis_mcp,
            test_polygon_rest_mcp, test_other_mcp_initialization,
            run_performance_test
        )
    logger.info("Successfully imported MCP test modules")
except ImportError as e:
    logger.error(f"Failed to import test modules: {e}")
    sys.exit(1)

# Test function mappings
TEST_FUNCTIONS = {
    "base": test_mcp_base,
    "yahoo": test_yahoo_finance_mcp,
    "trading": test_trading_mcp,
    "text": test_financial_text_mcp,
    "redis": test_redis_mcp,
    "polygon": test_polygon_rest_mcp,
    "others": test_other_mcp_initialization
}

def run_selected_tests(test_names: List[str], performance: bool = False, 
                      iterations: int = 1) -> MCPTestResults:
    """
    Run selected tests by name.
    
    Args:
        test_names: List of test names to run
        performance: Whether to run performance tests
        iterations: Number of times to run each test for performance measurement
        
    Returns:
        Test results object
    """
    results = MCPTestResults()
    
    # If "all" is specified, run all tests
    if "all" in test_names:
        test_names = list(TEST_FUNCTIONS.keys())
    
    logger.info(f"Preparing to run tests: {', '.join(test_names)}")
    
    # Run each specified test
    for test_name in test_names:
        if test_name in TEST_FUNCTIONS:
            logger.info(f"Running test: {test_name}")
            
            for i in range(iterations):
                if iterations > 1:
                    logger.info(f"  Iteration {i+1}/{iterations}")
                
                try:
                    start_time = time.time()
                    TEST_FUNCTIONS[test_name](results)
                    end_time = time.time()
                    
                    logger.info(f"Test {test_name} completed in {end_time - start_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error running test {test_name}: {e}")
                    results.record_test(
                        f"{test_name}_test_execution", False, 0.0, 
                        failure_reason=f"Unhandled exception: {str(e)}"
                    )
        else:
            logger.warning(f"Unknown test: {test_name}")
            results.add_warning("Test Selection", f"Unknown test name: {test_name}")
    
    return results


def list_available_tests():
    """List all available tests"""
    print("\nAvailable tests:")
    for name in TEST_FUNCTIONS.keys():
        print(f"  - {name}")
    print("\nSpecial options:")
    print("  - all: Run all available tests")
    print("\nExample usage:")
    print("  python run_mcp_tests.py base yahoo trading")
    print("  python run_mcp_tests.py all --iterations 3")


def run_mcp_installs_check() -> bool:
    """Check if all required MCP modules are installed"""
    required_modules = [
        "yfinance",
        "redis",
        "torch",
        "transformers",
        "alpaca-trade-api",
        "onnxruntime"
    ]
    
    missing_modules = []
    optional_modules = []
    
    print("\nChecking for required modules...")
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError:
            if module in ["torch", "transformers", "alpaca-trade-api", "onnxruntime"]:
                optional_modules.append(module)
                print(f"  ! {module} (optional, some tests may be limited)")
            else:
                missing_modules.append(module)
                print(f"  ✗ {module} (required)")
    
    if missing_modules:
        print("\nMissing required modules. Please install them with:")
        print(f"  pip install {' '.join(missing_modules)}\n")
        return False
        
    if optional_modules:
        print("\nSome optional modules are missing. For full testing capabilities, install them with:")
        print(f"  pip install {' '.join(optional_modules)}\n")
    
    return True


def main():
    """Main function to parse arguments and run tests"""
    parser = argparse.ArgumentParser(description="Run MCP tool tests")
    parser.add_argument("tests", nargs="*", default=["all"], 
                      help="Name of tests to run (default: all)")
    parser.add_argument("--list", action="store_true", 
                      help="List available tests and exit")
    parser.add_argument("--verbose", "-v", action="store_true", 
                      help="Enable verbose output")
    parser.add_argument("--performance", "-p", action="store_true", 
                      help="Run performance tests")
    parser.add_argument("--iterations", "-i", type=int, default=1,
                      help="Number of iterations for tests (for performance measurement)")
    parser.add_argument("--check-installs", action="store_true",
                      help="Check for required module installations")
    args = parser.parse_args()
    
    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("\n" + "="*80)
    print(f"MCP TOOLS TEST RUNNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Handle list option
    if args.list:
        list_available_tests()
        return
    
    # Check installations if requested
    if args.check_installs:
        if not run_mcp_installs_check():
            return
    
    # Print test plan
    tests_str = ", ".join(args.tests) if args.tests != ["all"] else "all tests"
    iterations_str = f" (x{args.iterations} iterations)" if args.iterations > 1 else ""
    performance_str = " with performance metrics" if args.performance else ""
    
    print(f"\nRunning {tests_str}{iterations_str}{performance_str}...\n")
    
    # Run selected tests
    start_time = time.time()
    results = run_selected_tests(args.tests, args.performance, args.iterations)
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
