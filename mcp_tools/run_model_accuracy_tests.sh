#!/bin/bash
# Run NextGen Model Accuracy Tests
# This script runs the model accuracy tests with proper environment setup

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    # More robust .env loading that handles special characters and spaces
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ $line =~ ^#.*$ ]] && continue
        [[ -z $line ]] && continue
        
        # Extract variable and value, handling spaces around equals sign
        if [[ $line =~ ^([A-Za-z0-9_]+)[[:space:]]*=[[:space:]]*(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Remove quotes if present
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            # Export the variable
            export "$key=$value"
            echo "  Loaded: $key"
        fi
    done < .env
else
    echo "Warning: .env file not found. Make sure environment variables are set."
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found"
    exit 1
fi

# Check for required packages
echo "Checking for required packages..."
python3 -c "import numpy, pandas, dotenv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install numpy pandas python-dotenv
fi

# Parse arguments
MODELS=""
CHECK_ENV=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check-env)
            CHECK_ENV=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            MODELS="$MODELS $1"
            shift
            ;;
    esac
done

# Build command
CMD="python3 mcp_tools/test_model_accuracy.py"

if [ "$CHECK_ENV" = true ]; then
    CMD="$CMD --check-env"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ -n "$MODELS" ]; then
    CMD="$CMD $MODELS"
fi

# Run the tests
echo "Running command: $CMD"
eval $CMD

# Store exit code
EXIT_CODE=$?

# Exit with the same code as the Python script
exit $EXIT_CODE
