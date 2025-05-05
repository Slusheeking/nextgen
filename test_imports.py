#!/usr/bin/env python3
"""
Test imports for all Python modules in the NextGen project.
This script attempts to import each Python module found in the project
and reports any modules that fail to import along with their error messages.
"""

import os
import sys
import importlib
import traceback
import subprocess
from pathlib import Path
from importlib import util
from typing import Dict, List, Tuple

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_colored(text: str, color: str) -> None:
    """Print text in color."""
    print(f"{color}{text}{RESET}")

def find_all_python_files() -> List[str]:
    """
    Find all Python files in the project using the 'find' command.
    
    Returns:
        List of Python file paths
    """
    # Run find command to get all Python files (excluding ta-lib)
    result = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", "*ta-lib*"],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Split the output into lines and remove the leading './' from each path
    python_files = [file.strip()[2:] for file in result.stdout.split("\n") if file.strip()]
    
    return python_files

def get_module_name(file_path: str) -> str:
    """
    Convert a file path to a module name.
    
    Args:
        file_path: Path to a Python file
    
    Returns:
        Module name in dot notation
    """
    # Remove .py extension
    if file_path.endswith(".py"):
        file_path = file_path[:-3]
    
    # Replace directory separators with dots
    module_name = file_path.replace(os.sep, ".")
    
    # Handle __init__.py files
    if module_name.endswith(".__init__"):
        module_name = module_name[:-9]
    
    return module_name

def test_import(module_name: str) -> Tuple[bool, str]:
    """
    Test importing a module and return success status and error message.
    
    Args:
        module_name: Name of the module to import
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as e:
        return False, f"{str(e)}\n{traceback.format_exc()}"

def test_all_imports() -> Dict[str, Tuple[bool, str]]:
    """
    Test importing all Python modules in the project.
    
    Returns:
        Dictionary mapping module names to (success, error_message) tuples
    """
    results = {}
    python_files = find_all_python_files()
    
    for file_path in python_files:
        module_name = get_module_name(file_path)
        success, error = test_import(module_name)
        results[module_name] = (success, error)
    
    return results

def analyze_errors(results: Dict[str, Tuple[bool, str]]) -> Dict[str, int]:
    """
    Analyze import errors to find common issues.
    
    Args:
        results: Dictionary mapping module names to (success, error_message) tuples
    
    Returns:
        Dictionary mapping error types to their occurrence count
    """
    error_types = {}
    
    for module_name, (success, error) in results.items():
        if not success:
            # Extract the first line of the error message
            error_first_line = error.split("\n")[0]
            
            if "ModuleNotFoundError" in error:
                # Extract the missing module name
                try:
                    missing_module = error.split("'")[1]
                    error_key = f"ModuleNotFoundError: {missing_module}"
                except IndexError:
                    error_key = "ModuleNotFoundError"
            else:
                error_key = error_first_line
            
            error_types[error_key] = error_types.get(error_key, 0) + 1
    
    return error_types

def main() -> None:
    """Main function to run the import tests."""
    print_colored("Testing imports for all Python modules...", BLUE)
    
    # Make sure the current directory is in the Python path
    sys.path.insert(0, os.getcwd())
    
    # Test all imports
    results = test_all_imports()
    
    # Count successes and failures
    success_count = sum(1 for success, _ in results.values() if success)
    failure_count = len(results) - success_count
    
    # Print summary
    print("\n" + "=" * 80)
    print_colored(f"SUMMARY: {success_count}/{len(results)} modules imported successfully", BLUE)
    print("=" * 80 + "\n")
    
    # Print successful imports
    if success_count > 0:
        print_colored("SUCCESSFUL IMPORTS:", GREEN)
        for module_name, (success, _) in sorted(results.items()):
            if success:
                print_colored(f"✓ {module_name}", GREEN)
        print()
    
    # Print failed imports
    if failure_count > 0:
        print_colored("FAILED IMPORTS:", RED)
        for module_name, (success, error) in sorted(results.items()):
            if not success:
                print_colored(f"✗ {module_name}", RED)
                for line in error.split("\n"):
                    if line.strip():
                        print(f"  {line}")
                print()
        
        # Analyze common errors
        error_types = analyze_errors(results)
        
        print_colored("COMMON ERROR TYPES:", YELLOW)
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print_colored(f"{count}x {error_type}", YELLOW)
        
        print("\nPossible solutions:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check for missing __init__.py files in package directories")
        print("3. Verify that the PYTHONPATH includes the project root directory")
        print("4. Check for circular imports in the codebase")
        print("5. Look for modules with the same name in different locations")
    
    sys.exit(1 if failure_count > 0 else 0)

if __name__ == "__main__":
    main()