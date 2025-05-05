# NextGen Model Accuracy Testing

This directory contains tools for testing the accuracy and functionality of NextGen models and their associated MCP tools.

## Overview

The testing framework allows you to:

1. Test each NextGen model individually or all models at once
2. Verify that all required MCP tools are working correctly
3. Test model accuracy using appropriate datasets
4. Test LLM integration for each model
5. Generate detailed reports on test results, including accuracy metrics

## Files

- `test_model_accuracy.py`: Main Python script for testing model accuracy
- `run_model_accuracy_tests.sh`: Shell script wrapper for running the tests with proper environment setup

## Available Models for Testing

- `fundamental`: Fundamental Analysis Model
- `sentiment`: Sentiment Analysis Model
- `market`: Market Analysis Model
- `risk`: Risk Assessment Model

## Prerequisites

Before running the tests, ensure you have:

1. Set up all required environment variables in your `.env` file:
   - `OPENROUTER_API_KEY`: For LLM access
   - `POLYGON_API_KEY`: For market data
   - `YAHOO_FINANCE_API_KEY`: For Yahoo Finance data (optional)
   - `REDIS_HOST` and `REDIS_PORT`: For Redis database access

2. Installed all required Python packages:
   - `numpy`, `pandas`, `python-dotenv`
   - `torch`, `transformers` (for LLM models)
   - `redis`, `yfinance`

## Usage

### Basic Usage

To test all models:

```bash
./run_model_accuracy_tests.sh
```

To test a specific model:

```bash
./run_model_accuracy_tests.sh fundamental
```

To test multiple specific models:

```bash
./run_model_accuracy_tests.sh fundamental sentiment
```

### Additional Options

- `--check-env`: Check environment setup and exit
- `--verbose` or `-v`: Enable verbose output

Examples:

```bash
# Check environment setup
./run_model_accuracy_tests.sh --check-env

# Test fundamental model with verbose output
./run_model_accuracy_tests.sh fundamental --verbose
```

## Test Results

The test results include:

- Overall pass/fail status for each test
- Detailed failure reasons for failed tests
- Performance metrics (execution time)
- Accuracy metrics where applicable
- Warnings for potential issues

## Extending the Tests

To add tests for a new model:

1. Create a new tester class that inherits from `ModelAccuracyTester`
2. Implement the required methods: `setup()`, `prepare_test_data()`, `verify_mcp_tools()`, etc.
3. Add specific test methods for your model
4. Add your model to the `MODEL_TESTERS` dictionary

## Troubleshooting

If tests are failing:

1. Check that all required environment variables are set correctly
2. Verify that all required Python packages are installed
3. Check that the Redis server is running
4. Look for specific error messages in the test output
5. Try running with the `--verbose` flag for more detailed output
