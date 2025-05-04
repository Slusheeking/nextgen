#!/usr/bin/env python3
"""
Simple test script to verify OpenRouter API connectivity.
"""

import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("openrouter_test")

# Load environment variables
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent
load_dotenv(dotenv_path=project_root / ".env")

# Get OpenRouter API key
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "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")

def test_openrouter_api():
    """Test direct API call to OpenRouter."""
    
    logger.info("Testing direct API call to OpenRouter...")
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "HTTP-Referer": "https://fingpt.ai",  # Optional
        "X-Title": "FinGPT Test",  # Optional
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What are the top 3 stocks to watch today and why?"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    try:
        logger.info("Sending request to OpenRouter API...")
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=payload)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Request completed in {duration:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("API call successful!")
            logger.info(f"Model used: {result.get('model', 'unknown')}")
            
            # Extract and print the response
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            logger.info(f"Response: {content}")
            
            return True, result
        else:
            logger.error(f"API call failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False, response.text
    except Exception as e:
        logger.error(f"Error making API call: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    logger.info("Starting OpenRouter API test...")
    success, result = test_openrouter_api()
    
    if success:
        logger.info("OpenRouter API test completed successfully!")
    else:
        logger.error("OpenRouter API test failed!")