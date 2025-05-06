"""End-to-end tests for monitoring and reporting."""

import json
import time
import logging
from collections import defaultdict

# Configure logging
LOG_FILE = "test_run.log"
logging.basicConfig(filename=LOG_FILE, level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(component)s - %(message)s')

# ANSI color codes
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_RESET = '\033[0m'

# Placeholder for collected monitoring data
# In a real scenario, this data would be collected from test runs
# and potentially stored in a file, database, or message queue.
# For this implementation, we'll use a simple dictionary structure
# to simulate receiving data.
monitoring_data = {
    "accuracy_inputs_outputs": [],  # List of {"input": ..., "output": ..., "expected_output": ..., "component": ...}
    "latency_measurements": [],     # List of {"component": ..., "start_time": ..., "end_time": ...}
    "error_logs": [],               # List of {"component": ..., "timestamp": ..., "message": ...}
    "warning_logs": []              # List of {"component": ..., "timestamp": ..., "message": ...}
}

def add_accuracy_data(input_data, output_data, expected_output, component):
    """Adds accuracy input/output data for a specific component."""
    monitoring_data["accuracy_inputs_outputs"].append({
        "input": input_data,
        "output": output_data,
        "expected_output": expected_output,
        "component": component
    })

def add_latency_measurement(component, start_time, end_time):
    """Adds a latency measurement for a specific component."""
    monitoring_data["latency_measurements"].append({
        "component": component,
        "start_time": start_time,
        "end_time": end_time
    })

def add_error_log(component, message):
    """Adds an error log entry for a specific component and logs to file."""
    log_entry = {
        "component": component,
        "timestamp": time.time(),
        "message": message
    }
    monitoring_data["error_logs"].append(log_entry)
    logging.error(message, extra={'component': component})

def add_warning_log(component, message):
    """Adds a warning log entry for a specific component and logs to file."""
    log_entry = {
        "component": component,
        "timestamp": time.time(),
        "message": message
    }
    monitoring_data["warning_logs"].append(log_entry)
    logging.warning(message, extra={'component': component})


def calculate_accuracy(data_list):
    """Calculates accuracy based on a list of input/output/expected data."""
    if not data_list:
        return 0.0

    correct_count = 0
    for item in data_list:
        # Simple equality check for demonstration.
        # More complex comparisons might be needed depending on data types.
        if item["output"] == item["expected_output"]:
            correct_count += 1

    return (correct_count / len(data_list)) * 100.0

def calculate_average_latency(latency_list):
    """Calculates the average latency from a list of latency measurements."""
    if not latency_list:
        return 0.0

    total_duration = 0
    for measurement in latency_list:
        total_duration += measurement["end_time"] - measurement["start_time"]

    return total_duration / len(latency_list)

def print_colored_logs():
    """Prints error and warning logs to the console with ANSI colors."""
    print("\n--- Error and Warning Logs (Terminal) ---")
    for log_entry in monitoring_data["error_logs"]:
        print(f"{COLOR_RED}ERROR - Component: {log_entry['component']} - Message: {log_entry['message']}{COLOR_RESET}")
    for log_entry in monitoring_data["warning_logs"]:
        print(f"{COLOR_YELLOW}WARNING - Component: {log_entry['component']} - Message: {log_entry['message']}{COLOR_RESET}")


def generate_monitoring_report():
    """Generates a comprehensive monitoring report."""
    report = {}

    # --- Accuracy Reporting ---
    report["accuracy"] = {}

    # Accuracy of individual models and MCP tools
    component_accuracy_data = defaultdict(list)
    for item in monitoring_data["accuracy_inputs_outputs"]:
        component_accuracy_data[item["component"]].append(item)

    for component, data in component_accuracy_data.items():
        report["accuracy"][f"{component}_accuracy"] = calculate_accuracy(data)

    # End-to-end accuracy (assuming 'end_to_end' component captures this)
    if "end_to_end" in component_accuracy_data:
        report["accuracy"]["end_to_end_accuracy"] = calculate_accuracy(component_accuracy_data["end_to_end"])
    else:
         report["accuracy"]["end_to_end_accuracy"] = "N/A (End-to-end accuracy data not available)"


    # --- Latency Reporting ---
    report["latency"] = {}

    # Latency of individual models and MCP tools
    component_latency_data = defaultdict(list)
    for item in monitoring_data["latency_measurements"]:
        component_latency_data[item["component"]].append(item)

    for component, data in component_latency_data.items():
        report["latency"][f"{component}_average_latency_seconds"] = calculate_average_latency(data)

    # Overall system flow latency (assuming 'system_flow' component captures this)
    if "system_flow" in component_latency_data:
        report["latency"]["system_flow_average_latency_seconds"] = calculate_average_latency(component_latency_data["system_flow"])
    else:
        report["latency"]["system_flow_average_latency_seconds"] = "N/A (System flow latency data not available)"


    # --- Error and Warning Reporting ---
    report["errors"] = monitoring_data.get("error_logs", [])
    report["warnings"] = monitoring_data.get("warning_logs", [])

    # Print colored logs to terminal
    print_colored_logs()

    # --- Report Summary ---
    report["summary"] = {
        "total_accuracy_checks": len(monitoring_data["accuracy_inputs_outputs"]),
        "total_latency_measurements": len(monitoring_data["latency_measurements"]),
        "total_errors": len(monitoring_data["error_logs"]),
        "total_warnings": len(monitoring_data["warning_logs"])
    }

    return report

# Example Usage (for demonstration purposes)
if __name__ == "__main__":
    # Simulate adding some data
    add_accuracy_data("input1", "output1", "output1", "model_A")
    add_accuracy_data("input2", "output2", "expected_output2", "model_A")
    add_accuracy_data("input3", "output3", "output3", "mcp_tool_X")
    add_accuracy_data("input4", "output4", "output4", "end_to_end")
    add_accuracy_data("input5", "output5", "expected_output5", "end_to_end")


    add_latency_measurement("model_A", time.time() - 0.1, time.time())
    add_latency_measurement("model_A", time.time() - 0.2, time.time())
    add_latency_measurement("mcp_tool_X", time.time() - 0.05, time.time())
    add_latency_measurement("system_flow", time.time() - 0.5, time.time())
    add_latency_measurement("system_flow", time.time() - 0.6, time.time())


    add_error_log("model_A", "Failed to process input")
    add_warning_log("model_A", "Potential issue with input format")
    add_error_log("mcp_tool_X", "Connection refused")
    add_warning_log("mcp_tool_X", "High latency detected")


    # Generate and print the report
    final_report = generate_monitoring_report()
    print("\n--- Full Monitoring Report (JSON) ---")
    print(json.dumps(final_report, indent=2))