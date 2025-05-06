#!/bin/bash
# NextGen Model Accuracy Test Runner
# This script runs the model accuracy tests using the Financial Phrasebank dataset

# Set script to exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="${PROJECT_ROOT}/financial_phrasebank.csv"
REPORT_PATH="${PROJECT_ROOT}/model_accuracy_report.json"
LOG_DIR="${PROJECT_ROOT}/logs"

# Parse command line arguments
VERBOSE=false
TEST_TYPE="all"

print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS] [TEST_TYPE]"
    echo
    echo "OPTIONS:"
    echo "  --verbose       Enable verbose output"
    echo "  --help          Display this help message"
    echo
    echo "TEST_TYPE:"
    echo "  all             Test all models and MCP tools (default)"
    echo "  sentiment       Test only sentiment analysis"
    echo "  select          Test only select model"
    echo "  financial_text  Test only financial text MCP"
    echo "  vector_store    Test only vector store MCP"
    echo "  redis           Test only Redis MCP"
    echo
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        all|sentiment|select|financial_text|vector_store|redis)
            TEST_TYPE=$1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option:${NC} $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${YELLOW}Dataset not found at:${NC} $DATASET_PATH"
    echo -e "${BLUE}Downloading Financial Phrasebank dataset...${NC}"
    
    # Try to download the dataset
    python3 -c "from datasets import load_dataset; dataset = load_dataset('takala/financial_phrasebank', 'sentences_allagree', split='train'); dataset.to_csv('$DATASET_PATH')"
    
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "${RED}Failed to download dataset.${NC}"
        exit 1
    else
        echo -e "${GREEN}Dataset downloaded successfully.${NC}"
    fi
else
    echo -e "${GREEN}Dataset found at:${NC} $DATASET_PATH"
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Run the test
echo -e "${BLUE}Running model accuracy tests...${NC}"
echo -e "${BLUE}Test type:${NC} $TEST_TYPE"

# Build command
CMD="python3 ${PROJECT_ROOT}/scripts/test_model_accuracy.py"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

CMD="$CMD --report-file $REPORT_PATH"

# Add test type if not "all"
if [ "$TEST_TYPE" != "all" ]; then
    CMD="$CMD --test-type $TEST_TYPE"
fi

# Run the command
echo -e "${BLUE}Executing:${NC} $CMD"
eval "$CMD"

# Check if report was generated
if [ -f "$REPORT_PATH" ]; then
    echo -e "${GREEN}Test completed successfully.${NC}"
    echo -e "${GREEN}Report saved to:${NC} $REPORT_PATH"
    
    # Extract and display key metrics
    echo -e "${BLUE}Extracting key metrics from report...${NC}"
    python3 -c "
import json
import sys
try:
    with open('$REPORT_PATH', 'r') as f:
        data = json.load(f)
    
    if 'overall_metrics' in data:
        metrics = data['overall_metrics']
        print(f\"\\n===== KEY METRICS =====\")
        print(f\"Average model accuracy: {metrics.get('average_model_accuracy', 'N/A'):.4f}\")
        print(f\"Models tested: {metrics.get('models_tested', 'N/A')}\")
        print(f\"MCP tools tested: {metrics.get('mcp_tools_tested', 'N/A')}\")
        print(f\"Error count: {metrics.get('error_count', 'N/A')}\")
        print(f\"Average CPU usage: {metrics.get('average_cpu_usage', 'N/A'):.2f}%\")
        print(f\"Average memory usage: {metrics.get('average_memory_usage', 'N/A'):.2f}%\")
        print(f\"=======================\\n\")
    else:
        print(\"\\nNo overall metrics found in report.\\n\")
except Exception as e:
    print(f\"Error extracting metrics: {e}\")
    sys.exit(1)
"
    
    # Find the latest log file
    LATEST_LOG=$(ls -t "$LOG_DIR"/model_accuracy_test_*.log 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
        echo -e "${GREEN}Log file:${NC} $LATEST_LOG"
        
        # Extract errors and warnings
        echo -e "${YELLOW}Errors and warnings from log:${NC}"
        grep -E 'ERROR|WARNING' "$LATEST_LOG" | tail -n 10
    fi
else
    echo -e "${RED}Test failed. No report generated.${NC}"
    exit 1
fi

echo -e "${GREEN}Model accuracy test completed.${NC}"
