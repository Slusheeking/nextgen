"""Script to run all end-to-end tests with environment-based configuration."""
import argparse
import sys
import os
import time
import json
import logging
import dotenv

# Add the parent directory to the Python path so we can import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

# Import test modules
import test_data_generation
import test_models
import test_mcp_tools
import test_integrated_system
# Import monitoring module for reporting
import test_monitoring_and_reporting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tests(data_source):
    """
    Runs all end-to-end tests with the specified data source, focusing on integrated model+MCP testing.

    Sets environment variables for the data source and database connections. Orchestrates
    the execution of test modules in a logical sequence from individual components to
    integrated systems. Collects monitoring data and generates a final report.

    Args:
        data_source (str): The data source to use ('live', 'synthetic', 'downloaded').
    """
    logger.info(f"Starting end-to-end tests with data source: {data_source}")

    # --- Setup ---
    logger.info("\n--- Test Environment Setup ---")
    
    # Load environment variables from .env file
    dotenv.load_dotenv()
    logger.info("Loaded environment variables from .env file")
    
    # Set environment variable for data source
    os.environ['E2E_DATA_SOURCE'] = data_source
    logger.info(f"Environment variable E2E_DATA_SOURCE set to: {data_source}")
    
    # Check for required environment variables
    required_envs = ["REDIS_HOST", "CHROMADB_HOST"]
    missing_envs = [env for env in required_envs if not os.getenv(env)]
    
    if missing_envs:
        logger.warning(f"Missing required environment variables: {missing_envs}")
        logger.warning("Using default values for missing environment variables")
        
        # Set default values for required environment variables
        if "REDIS_HOST" in missing_envs:
            os.environ["REDIS_HOST"] = "localhost"
        if "CHROMADB_HOST" in missing_envs:
            os.environ["CHROMADB_HOST"] = "localhost"
    
    # Initialize test environment
    setup_start_time = time.time()
    logger.info("Initializing test environment...")
    
    # Wait for services to be ready if needed
    if data_source != 'synthetic':
        logger.info("Waiting for services to be ready...")
        time.sleep(1)  # Simulated wait, in a real scenario we would check service health
        
    setup_end_time = time.time()
    logger.info(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds.")


    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # --- Test Execution ---
    logger.info("\n--- Running Test Suites ---")

    # First run: Individual component tests to ensure basic functionality
    logger.info("Adding individual component tests...")
    suite.addTests(loader.loadTestsFromModule(test_models))
    suite.addTests(loader.loadTestsFromModule(test_mcp_tools))
    
    # Second run: Integrated system test (model+MCP combinations)
    logger.info("Adding integrated system tests for model+MCP combinations...")
    suite.addTests(loader.loadTestsFromModule(test_integrated_system))

    # Note: test_monitoring_and_reporting is used for reporting, not as a test suite to run here.
    logger.info("Note: test_monitoring_and_reporting will be used for output reporting")

    # Execute all tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_run_start_time = time.time()
    result = runner.run(suite)
    test_run_end_time = time.time()
    test_run_duration = test_run_end_time - test_run_start_time
    logger.info(f"\nTest execution completed in {test_run_duration:.2f} seconds.")
    
    # Update test runtime in monitoring data
    if hasattr(test_monitoring_and_reporting, 'monitoring_data') and isinstance(test_monitoring_and_reporting.monitoring_data, dict):
        if "test_metadata" in test_monitoring_and_reporting.monitoring_data:
            test_monitoring_and_reporting.monitoring_data["test_metadata"]["total_runtime"] = test_run_duration

    # --- Monitoring and Reporting ---
    logger.info("\n--- Generating Monitoring Report ---")
    # Generate final report from monitoring data
    final_report = test_monitoring_and_reporting.generate_monitoring_report()

    logger.info("\n--- Final Monitoring Report ---")
    # Print the report and save to file
    print(json.dumps(final_report, indent=2))
    
    # Save report to file
    report_path = "e2e_test_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        logger.info(f"Test report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save test report: {e}")

    # --- Teardown ---
    logger.info("\n--- Test Environment Teardown ---")
    # Clean up environment variable
    if 'E2E_DATA_SOURCE' in os.environ:
        del os.environ['E2E_DATA_SOURCE']
        logger.info("Environment variable E2E_DATA_SOURCE removed.")
    
    # Perform cleanup operations
    teardown_start_time = time.time()
    logger.info("Cleaning up test resources...")
    
    # No need to actually stop services in this environment
    # In a real scenario, we would stop/clean up any services started for testing
    
    teardown_end_time = time.time()
    logger.info(f"Teardown completed in {teardown_end_time - teardown_start_time:.2f} seconds.")


    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end tests with a specified data source.")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=['live', 'synthetic', 'downloaded'],
        required=True,
        help="The data source to use for tests: 'live', 'synthetic', or 'downloaded'."
    )

    args = parser.parse_args()

    # Clear previous monitoring data and initialize with testing metadata
    current_time = time.time()
    test_monitoring_and_reporting.monitoring_data = {
        "accuracy_inputs_outputs": [],
        "latency_measurements": [],
        "error_logs": [],
        "warning_logs": [],  # This key was missing
        "test_metadata": {
            "data_source": args.data_source,
            "timestamp": current_time,
            "total_runtime": 0,  # Will be updated after test completion
            "test_modules": [
                "test_models (individual models with MCP dependencies)",
                "test_mcp_tools (individual MCP tools)",
                "test_integrated_system (integrated model+MCP combinations)"
            ]
        }
    }
    logger.info("Initialized monitoring data with test metadata.")


    if not run_tests(args.data_source):
        sys.exit(1)

    sys.exit(0)