#!/usr/bin/env python3
"""
Setup Monitoring Script

This script integrates Prometheus metrics and Loki logging into
specified directories of the NextGen FinGPT project.
"""

import os
import re
import sys
from pathlib import Path

# Directories to integrate monitoring into
DIRECTORIES = [
    "mcp_tools/data_mcp",
    "fingpt/autogen_orchestrator",
    "alpaca",
    "fingpt/nextgen_selection",
    "mcp_tools/execution_mcp"
]

# Import statement to add
IMPORT_STATEMENT = """
# Import monitoring utilities
from prometheus.prometheus_loki_utils import setup_monitoring
"""

# Monitoring setup code to add
MONITORING_SETUP = """
# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="{service_name}",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={{"component": "{component}"}}
)
"""

def get_service_name(file_path):
    """
    Generate a service name from the file path.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: Service name.
    """
    # Extract the filename without extension
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]
    
    # Convert to kebab-case
    service_name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', filename).lower()
    service_name = re.sub(r'_', '-', service_name)
    
    # Get the directory name
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Combine directory and filename
    return f"{dir_name}-{service_name}"

def get_component_name(file_path):
    """
    Generate a component name from the file path.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: Component name.
    """
    # Get the directory structure
    parts = file_path.split(os.path.sep)
    
    # Use the top-level directory and the immediate parent directory
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[-2]}"
    else:
        return parts[0]

def add_monitoring_to_file(file_path):
    """
    Add monitoring imports and setup to a Python file.
    
    Args:
        file_path (str): Path to the Python file.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if monitoring is already set up
        if "setup_monitoring" in content:
            print(f"Monitoring already set up in {file_path}")
            return True
        
        # Find the last import statement
        import_pattern = re.compile(r'^(import|from)\s+.*$', re.MULTILINE)
        imports = list(import_pattern.finditer(content))
        
        if imports:
            last_import = imports[-1]
            insert_pos = last_import.end()
            
            # Insert the import statement after the last import
            content = content[:insert_pos] + IMPORT_STATEMENT + content[insert_pos:]
        else:
            # If no imports found, add at the beginning after any docstring
            docstring_end = content.find('"""', content.find('"""') + 3) if '"""' in content else -1
            if docstring_end != -1:
                insert_pos = docstring_end + 3
                # Add a newline if not already present
                if content[insert_pos:insert_pos+2] != '\n\n':
                    content = content[:insert_pos] + '\n\n' + content[insert_pos:]
                    insert_pos += 2
            else:
                insert_pos = 0
            
            content = content[:insert_pos] + IMPORT_STATEMENT + content[insert_pos:]
        
        # Find a good place to insert the monitoring setup
        # Look for class or function definitions
        class_pattern = re.compile(r'^class\s+\w+', re.MULTILINE)
        func_pattern = re.compile(r'^def\s+\w+', re.MULTILINE)
        
        class_matches = list(class_pattern.finditer(content))
        func_matches = list(func_pattern.finditer(content))
        
        # Determine service and component names
        service_name = get_service_name(file_path)
        component_name = get_component_name(file_path)
        
        # Format the monitoring setup code
        setup_code = MONITORING_SETUP.format(
            service_name=service_name,
            component=component_name
        )
        
        if class_matches:
            # Insert before the first class definition
            insert_pos = class_matches[0].start()
            # Ensure there are newlines before the class definition
            content = content[:insert_pos] + setup_code + '\n\n' + content[insert_pos:]
        elif func_matches:
            # Insert before the first function definition
            insert_pos = func_matches[0].start()
            # Ensure there are newlines before the function definition
            content = content[:insert_pos] + setup_code + '\n\n' + content[insert_pos:]
        else:
            # If no class or function definitions, add at the end of the file
            content += '\n\n' + setup_code
        
        # Write the modified content back to the file
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Added monitoring to {file_path}")
        return True
    
    except Exception as e:
        print(f"Error adding monitoring to {file_path}: {e}")
        return False

def process_directory(directory):
    """
    Process all Python files in a directory.
    
    Args:
        directory (str): Directory to process.
    
    Returns:
        int: Number of files processed.
    """
    count = 0
    
    # Get all Python files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                if add_monitoring_to_file(file_path):
                    count += 1
    
    return count

def main():
    """Main function."""
    # Get the project root directory
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    
    # Change to the project root directory
    os.chdir(project_root)
    
    print(f"Setting up monitoring in {project_root}")
    
    # Process each directory
    total_processed = 0
    for directory in DIRECTORIES:
        print(f"\nProcessing directory: {directory}")
        processed = process_directory(directory)
        total_processed += processed
        print(f"Processed {processed} files in {directory}")
    
    print(f"\nTotal files processed: {total_processed}")
    print("Monitoring setup complete!")

if __name__ == "__main__":
    main()
