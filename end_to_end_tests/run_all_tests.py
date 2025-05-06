"""Script to run all end-to-end tests."""
import argparse
import sys
import unittest
import os # Import os to set environment variable
import time # Import time for basic timing/setup
import json # Import json for report printing

# Import test modules
from end_to_end_tests import test_data_generation
from end_to_end_tests import test_models
from end_to_end_tests import test_mcp_tools
from end_to_end_tests import test_integrated_system
# Import monitoring module for reporting
from end_to_end_tests import test_monitoring_and_reporting

def run_tests(data_source):
    """
    Runs all end-to-end tests with the specified data source.

    Sets an environment variable for the data source and orchestrates
    the execution of test modules in a logical sequence. Collects
    monitoring data and generates a final report.

    Args:
        data_source (str): The data source to use ('live', 'synthetic', 'downloaded').
    """
    print(f"Starting end-to-end tests with data source: {data_source}")

    # --- Setup ---
    print("\n--- Test Environment Setup ---")
    # Set environment variable for data source.
    # Note: Test modules need to be updated to read and use this variable.
    # Due to constraints, we cannot modify other test files, so this variable
    # serves as the mechanism for passing the data source information.
    os.environ['E2E_DATA_SOURCE'] = data_source
    print(f"Environment variable E2E_DATA_SOURCE set to: {data_source}")
    # Placeholder for other setup logic (e.g., starting services)
    print("Placeholder: Starting necessary services or initializing components.")
    setup_start_time = time.time()
    # Simulate some setup time
    time.sleep(1)
    setup_end_time = time.time()
    print(f"Setup completed in {setup_end_time - setup_start_time:.2f} seconds.")


    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # --- Test Execution ---
    print("\n--- Running Test Suites ---")

    # Add tests from each module in sequence
    # Data generation (if synthetic data source is selected)
    if data_source == 'synthetic':
         print("Adding data generation tests...")
         suite.addTests(loader.loadTestsFromModule(test_data_generation))

    # Individual component tests
    print("Adding individual component tests...")
    suite.addTests(loader.loadTestsFromModule(test_models))
    suite.addTests(loader.loadTestsFromModule(test_mcp_tools))

    # Integrated system test
    print("Adding integrated system tests...")
    suite.addTests(loader.loadTestsFromModule(test_integrated_system))

    # Note: test_monitoring_and_reporting is used for reporting, not as a test suite to run here.

    runner = unittest.TextTestRunner(verbosity=2)
    test_run_start_time = time.time()
    result = runner.run(suite)
    test_run_end_time = time.time()
    print(f"\nTest execution completed in {test_run_end_time - test_run_start_time:.2f} seconds.")


    # --- Monitoring and Reporting ---
    print("\n--- Generating Monitoring Report ---")
    # Assuming test modules have called monitoring functions during their execution.
    final_report = test_monitoring_and_reporting.generate_monitoring_report()

    print("\n--- Final Monitoring Report ---")
    # Print the report (or save it to a file)
    print(json.dumps(final_report, indent=2))


    # --- Teardown ---
    print("\n--- Test Environment Teardown ---")
    # Clean up environment variable
    if 'E2E_DATA_SOURCE' in os.environ:
        del os.environ['E2E_DATA_SOURCE']
        print("Environment variable E2E_DATA_SOURCE removed.")
    # Placeholder for other teardown logic (e.g., stopping services)
    print("Placeholder: Stopping services or cleaning up resources.")
    teardown_start_time = time.time()
    # Simulate some teardown time
    time.sleep(0.5)
    teardown_end_time = time.time()
    print(f"Teardown completed in {teardown_end_time - teardown_start_time:.2f} seconds.")


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

    # Clear previous monitoring data before running tests
    # This assumes test_monitoring_and_reporting uses a global state.
    # A more robust solution would involve passing a monitoring object,
    # but due to constraints, clearing global state is used as a workaround.
    test_monitoring_and_reporting.monitoring_data = {
        "accuracy_inputs_outputs": [],
        "latency_measurements": [],
        "error_logs": []
    }
    print("Cleared previous monitoring data.")


    if not run_tests(args.data_source):
        sys.exit(1)

    sys.exit(0)