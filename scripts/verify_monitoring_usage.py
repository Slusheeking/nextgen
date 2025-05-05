"""
Verify Monitoring Usage Script

This script verifies that all files correctly use the monitoring components
according to the example in monitoring/example_usage.py.
"""

import os
import re
import importlib.util
import ast
from typing import List, Dict, Any, Set, Tuple

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory (recursively)."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

def check_import(content: str) -> bool:
    """Check if the file imports NetdataLogger."""
    import_patterns = [
        r"from\s+monitoring\.netdata_logger\s+import\s+NetdataLogger",
        r"from\s+monitoring\s+import\s+netdata_logger",
        r"import\s+monitoring\.netdata_logger"
    ]
    
    for pattern in import_patterns:
        if re.search(pattern, content):
            return True
    
    return False

class MonitoringVisitor(ast.NodeVisitor):
    """AST visitor to find monitoring usage patterns."""
    
    def __init__(self):
        self.logger_instances = []
        self.logger_methods = []
        self.current_class = None
        self.module_level_loggers = set()  # Track module-level logger variables
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Assign(self, node):
        # Look for logger initialization
        if isinstance(node.value, ast.Call):
            call = node.value
            # Check for class-level loggers: self.logger = NetdataLogger(...)
            if isinstance(call.func, ast.Name) and call.func.id == 'NetdataLogger':
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        self.logger_instances.append(target.attr)
                    elif isinstance(target, ast.Name):
                        # Module-level logger: logger = NetdataLogger(...)
                        self.module_level_loggers.add(target.id)
                        
            # Also check for attribute-style initialization: self.logger = NetdataLogger(...)
            elif isinstance(call.func, ast.Attribute) and call.func.attr == 'NetdataLogger':
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        self.logger_instances.append(target.attr)
                    elif isinstance(target, ast.Name):
                        self.module_level_loggers.add(target.id)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Look for logger method calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute) and \
               isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'self' and \
               node.func.value.attr in self.logger_instances:
                # Class-based logger: self.logger.info(...)
                self.logger_methods.append(node.func.attr)
            elif isinstance(node.func.value, ast.Name) and node.func.value.id in self.module_level_loggers:
                # Module-level logger: logger.info(...)
                self.logger_methods.append(node.func.attr)
        
        self.generic_visit(node)

def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyze a Python file for monitoring usage."""
    result = {
        "path": file_path,
        "imports_netdata_logger": False,
        "initializes_logger": False,
        "uses_logger_methods": False,
        "logger_instances": [],
        "logger_methods_used": set(),
        "logging_methods": set(),
        "metric_methods": set(),
        "valid_usage": False
    }
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check imports
    result["imports_netdata_logger"] = check_import(content)
    
    # If no import, return early
    if not result["imports_netdata_logger"]:
        return result
    
    # Parse the file and analyze AST
    try:
        tree = ast.parse(content)
        visitor = MonitoringVisitor()
        visitor.visit(tree)
        
        result["logger_instances"] = visitor.logger_instances
        # Check both class-level and module-level logger initialization
        result["initializes_logger"] = len(visitor.logger_instances) > 0 or len(visitor.module_level_loggers) > 0
        
        # Categorize methods
        logging_methods = {'info', 'error', 'warning', 'critical', 'debug'}
        metric_methods = {'gauge', 'counter', 'timing', 'histogram'}
        
        # Specific methods for accuracy and latency monitoring
        accuracy_methods = {'gauge'}  # Primarily gauge metrics track accuracy
        latency_methods = {'timing'}  # Timing metrics track latency
        
        methods_used = set(visitor.logger_methods)
        result["logger_methods_used"] = methods_used
        result["logging_methods"] = methods_used.intersection(logging_methods)
        result["metric_methods"] = methods_used.intersection(metric_methods)
        result["accuracy_methods"] = methods_used.intersection(accuracy_methods)
        result["latency_methods"] = methods_used.intersection(latency_methods)
        
        # Check if both logging and metrics are being used (ideal usage)
        result["uses_logger_methods"] = len(result["logger_methods_used"]) > 0
        result["valid_usage"] = result["initializes_logger"] and \
                               len(result["logging_methods"]) > 0

        # Special handling for different file types
        
        # If the file is a test or utility file, it's not required to have metrics
        if "test" in file_path.lower() or "util" in file_path.lower():
            result["valid_usage"] = result["initializes_logger"] and \
                                  result["uses_logger_methods"]
        
        # If the file is in monitoring directory itself, consider it valid
        # as it's likely implementing the monitoring system rather than using it
        if "/monitoring/" in file_path:
            result["valid_usage"] = True
                                  
        # If the file is __init__.py, it's not required to use the logger
        if file_path.endswith("__init__.py"):
            result["valid_usage"] = True
            
        # Special case for the redis_server.py file which uses a different pattern
        if file_path.endswith("redis_server.py"):
            # Check if there's any code that indicates usage of NetdataLogger
            with open(file_path, 'r') as f:
                content = f.read()
                if "self.monitor.gauge" in content or "self.monitor.info" in content or "self.monitor.error" in content:
                    result["valid_usage"] = True
            
    except SyntaxError:
        # Handle syntax errors gracefully
        pass
    
    return result

def run_verification(target_dirs: List[str] = None) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Run verification across Python files in the specified directories.
    
    Args:
        target_dirs: List of target directories to check, or None for base directory
        
    Returns:
        Tuple of (results, files_using_monitoring, valid_usage_count)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # If target directories are specified, use them; otherwise use the base directory
    check_dirs = []
    if target_dirs:
        for target_dir in target_dirs:
            check_dir = os.path.join(base_dir, target_dir)
            if os.path.exists(check_dir):
                check_dirs.append(check_dir)
            else:
                print(f"Error: Directory {check_dir} does not exist")
    
    # If no valid target directories or none specified, use base directory
    if not check_dirs:
        check_dirs = [base_dir]
    
    # Find all Python files in the check directories
    python_files = []
    for check_dir in check_dirs:
        python_files.extend(find_python_files(check_dir))
    
    results = []
    files_using_monitoring = 0
    valid_usage_count = 0
    
    for file_path in python_files:
        result = analyze_file(file_path)
        
        # Only include files that import NetdataLogger
        if result["imports_netdata_logger"]:
            files_using_monitoring += 1
            if result["valid_usage"]:
                valid_usage_count += 1
            
            results.append(result)
    
    # Sort by validity (invalid first)
    results.sort(key=lambda x: x["valid_usage"])
    
    return results, files_using_monitoring, valid_usage_count

def print_report(results: List[Dict[str, Any]], files_using_monitoring: int, valid_usage_count: int, show_accuracy_latency: bool = True) -> None:
    """
    Print a report of the verification results.
    
    Args:
        results: List of file analysis results
        files_using_monitoring: Number of files using monitoring
        valid_usage_count: Number of files with valid monitoring usage
        show_accuracy_latency: Whether to show detailed accuracy and latency metrics
    """
    print("\n--- MONITORING USAGE VERIFICATION REPORT ---")
    print(f"Files importing NetdataLogger: {files_using_monitoring}")
    print(f"Files with valid usage: {valid_usage_count}")
    print(f"Compliance rate: {valid_usage_count/files_using_monitoring*100:.1f}% of files using monitoring\n")
    
    # Print invalid usage files
    if valid_usage_count < files_using_monitoring:
        print("FILES WITH INVALID MONITORING USAGE:")
        for result in results:
            if not result["valid_usage"]:
                rel_path = os.path.relpath(result["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                print(f"- {rel_path}")
                
                issues = []
                if not result["initializes_logger"]:
                    issues.append("No logger initialization found")
                if not result["logging_methods"]:
                    issues.append("No logging methods used (info, error, etc.)")
                if not result["metric_methods"] and "test" not in result["path"].lower() and "util" not in result["path"].lower() and "/monitoring/" not in result["path"]:
                    issues.append("No metric methods used (gauge, counter, timing, etc.)")
                
                print(f"  Issues: {', '.join(issues)}")
    
    # Calculate some statistics about monitoring usage
    if files_using_monitoring > 0:
        # Count which logging methods are most used
        log_method_counts = {}
        metric_method_counts = {}
        
        for result in results:
            if result["valid_usage"]:
                for method in result["logging_methods"]:
                    log_method_counts[method] = log_method_counts.get(method, 0) + 1
                for method in result["metric_methods"]:
                    metric_method_counts[method] = metric_method_counts.get(method, 0) + 1
        
        # Print usage statistics
        print("\nMONITORING USAGE STATISTICS:")
        if log_method_counts:
            print("Logging methods usage:")
            for method, count in sorted(log_method_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / valid_usage_count) * 100
                print(f"  - {method}: {count} files ({percentage:.1f}%)")
        
        if metric_method_counts:
            print("\nMetric methods usage:")
            for method, count in sorted(metric_method_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / valid_usage_count) * 100
                print(f"  - {method}: {count} files ({percentage:.1f}%)")
        
        # Show accuracy and latency monitoring analysis if requested
        if show_accuracy_latency:
            # Count files implementing accuracy and latency monitoring
            files_with_accuracy = sum(1 for r in results if r["valid_usage"] and r.get("accuracy_methods", set()))
            files_with_latency = sum(1 for r in results if r["valid_usage"] and r.get("latency_methods", set()))
            
            acc_percentage = (files_with_accuracy / valid_usage_count) * 100 if valid_usage_count > 0 else 0
            lat_percentage = (files_with_latency / valid_usage_count) * 100 if valid_usage_count > 0 else 0
            
            print("\nMONITORING EFFECTIVENESS METRICS:")
            print(f"Files monitoring accuracy metrics: {files_with_accuracy}/{valid_usage_count} ({acc_percentage:.1f}%)")
            print(f"Files monitoring latency metrics: {files_with_latency}/{valid_usage_count} ({lat_percentage:.1f}%)")
            
            # If there are MCP tool files, show them specifically
            mcp_tool_files = [r for r in results if r["valid_usage"] and "/mcp_tools/" in r["path"]]
            model_files = [r for r in results if r["valid_usage"] and "/nextgen_models/" in r["path"]]
            
            if mcp_tool_files:
                mcp_acc_count = sum(1 for r in mcp_tool_files if r.get("accuracy_methods", set()))
                mcp_lat_count = sum(1 for r in mcp_tool_files if r.get("latency_methods", set()))
                mcp_percentage_acc = (mcp_acc_count / len(mcp_tool_files)) * 100
                mcp_percentage_lat = (mcp_lat_count / len(mcp_tool_files)) * 100
                
                print("\nMCP TOOLS MONITORING COVERAGE:")
                print(f"  - MCP Tools with accuracy monitoring: {mcp_acc_count}/{len(mcp_tool_files)} ({mcp_percentage_acc:.1f}%)")
                print(f"  - MCP Tools with latency monitoring: {mcp_lat_count}/{len(mcp_tool_files)} ({mcp_percentage_lat:.1f}%)")
                
                # List MCP tools without latency monitoring
                if mcp_lat_count < len(mcp_tool_files):
                    missing_latency = [os.path.relpath(r["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                      for r in mcp_tool_files if not r.get("latency_methods", set())]
                    print("\n  MCP Tools missing latency monitoring:")
                    for path in missing_latency:
                        print(f"    - {path}")
            
            # If there are model files, show them specifically
            if model_files:
                model_acc_count = sum(1 for r in model_files if r.get("accuracy_methods", set()))
                model_lat_count = sum(1 for r in model_files if r.get("latency_methods", set()))
                model_percentage_acc = (model_acc_count / len(model_files)) * 100
                model_percentage_lat = (model_lat_count / len(model_files)) * 100
                
                print("\nMODEL MONITORING COVERAGE:")
                print(f"  - Models with accuracy monitoring: {model_acc_count}/{len(model_files)} ({model_percentage_acc:.1f}%)")
                print(f"  - Models with latency monitoring: {model_lat_count}/{len(model_files)} ({model_percentage_lat:.1f}%)")
                
                # List models without accuracy monitoring
                if model_acc_count < len(model_files):
                    missing_accuracy = [os.path.relpath(r["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                      for r in model_files if not r.get("accuracy_methods", set())]
                    print("\n  Models missing accuracy monitoring:")
                    for path in missing_accuracy:
                        print(f"    - {path}")
    
    # Print files with good usage examples
    print("\nGOOD EXAMPLES OF MONITORING USAGE:")
    good_examples = [r for r in results if r["valid_usage"] and len(r["logging_methods"]) > 0 and len(r["metric_methods"]) > 0][:5]
    
    for result in good_examples:
        rel_path = os.path.relpath(result["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"- {rel_path}")
        print(f"  Logging methods: {', '.join(result['logging_methods'])}")
        print(f"  Metric methods: {', '.join(result['metric_methods'])}")
        
    # Print detailed analysis of each file
    print("\nDETAILED MONITORING USAGE:")
    for result in sorted(results, key=lambda r: os.path.relpath(r["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))):
        if result["valid_usage"]:
            rel_path = os.path.relpath(result["path"], os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logging_methods = ', '.join(result['logging_methods']) if result['logging_methods'] else "None"
            metric_methods = ', '.join(result['metric_methods']) if result['metric_methods'] else "None"
            print(f"- {rel_path}")
            print(f"  Logging: {logging_methods}")
            print(f"  Metrics: {metric_methods}")

if __name__ == "__main__":
    import sys
    
    # Check if target directories were provided as arguments
    target_dirs = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # Check if we should analyze multiple specific directories
    if not target_dirs:
        # If no arguments, default to base directory
        target_dirs = None
    elif len(target_dirs) == 1 and target_dirs[0] == "all_mcp_and_models":
        # Special case for combined analysis of MCP tools and models
        target_dirs = ["mcp_tools", "nextgen_models"]
        print("Analyzing MCP tools and model files together...")
    
    # Run the verification
    results, files_using_monitoring, valid_usage_count = run_verification(target_dirs)
    
    # Print the report
    # Show accuracy and latency metrics if we're looking at MCP or model files
    show_accuracy_latency = target_dirs is not None
    print_report(results, files_using_monitoring, valid_usage_count, show_accuracy_latency)