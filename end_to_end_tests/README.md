# End-to-End Test Framework

## 1. Introduction

This end-to-end test framework is designed to validate the entire system pipeline, from data handling and processing through individual components (models and MCP tools) to the integrated system and final monitoring and reporting. It ensures that all parts of the system work together correctly and produce the expected results under various data conditions.

## 2. How to Use

The tests are executed using the `run_all_tests.py` script. You can specify the data source for the tests using a command-line argument.

To run the tests, navigate to the `end_to_end_tests` directory in your terminal and execute the script with the desired data source:

```bash
python run_all_tests.py --data-source [live|synthetic|downloaded]
```

Replace `[live|synthetic|downloaded]` with one of the following options:

-   `live`: Use live data for testing.
-   `synthetic`: Use synthetically generated data for testing.
-   `downloaded`: Use pre-downloaded data for testing.

For example, to run tests using synthetic data:

```bash
python run_all_tests.py --data-source synthetic
```

## 3. How it Works

The `end_to_end_tests` directory contains several Python files, each responsible for a specific part of the testing process:

-   `run_all_tests.py`: This is the main entry point for the test framework. It parses command-line arguments, orchestrates the execution of the different test stages, and aggregates results.
-   `test_data_generation.py`: Handles the setup and generation or loading of test data based on the specified data source (`live`, `synthetic`, or `downloaded`).
-   `test_models.py`: Contains tests specifically for validating the functionality and output of the individual machine learning models used in the system.
-   `test_mcp_tools.py`: Contains tests for verifying the correct operation and integration of the various MCP (Model Context Protocol) tools.
-   `test_integrated_system.py`: Focuses on testing the complete system flow, ensuring that data is correctly passed between components and that the final output is as expected when all parts are working together.
-   `test_monitoring_and_reporting.py`: Contains tests to verify that the monitoring and reporting components correctly capture system metrics, logs, and generate reports.

## 4. Test Flow

When `run_all_tests.py` is executed, the tests follow a specific sequence to ensure dependencies are met and the system is tested progressively. The general flow is as follows:

1.  **Start:** The `run_all_tests.py` script is initiated with the specified data source.
2.  **Data Handling:** `test_data_generation.py` prepares the test data according to the chosen source.
3.  **Individual Tests:** Tests for individual components (`test_models.py` and `test_mcp_tools.py`) are run to ensure they function correctly in isolation.
4.  **Integrated System Test:** `test_integrated_system.py` runs tests that involve the interaction of multiple components, simulating the full system pipeline.
5.  **Monitoring and Reporting:** `test_monitoring_and_reporting.py` verifies that monitoring data is collected and reports are generated correctly based on the test execution.
6.  **End:** The test execution concludes.

Here is a Mermaid diagram illustrating the test flow:

```mermaid
graph TD
    A[Start] --> B{Run run_all_tests.py};
    B --> C[Data Handling<br>test_data_generation.py];
    C --> D[Individual Tests];
    D --> E[Test Models<br>test_models.py];
    D --> F[Test MCP Tools<br>test_mcp_tools.py];
    E --> G[Integrated System Test<br>test_integrated_system.py];
    F --> G;
    G --> H[Monitoring and Reporting<br>test_monitoring_and_reporting.py];
    H --> I[End];