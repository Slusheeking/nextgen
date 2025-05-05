#!/bin/bash

# NextGen Trading System Run Script
# This script manages starting and shutting down the entire NextGen trading system,
# including Redis server, monitoring components, and MCP servers.

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define log directory
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Define PID file location
PID_DIR="$LOGS_DIR/pids"
mkdir -p "$PID_DIR"

# Function to display the script usage
usage() {
  echo -e "${BLUE}Usage:${NC} $0 [start|stop|status|restart]"
  echo ""
  echo -e "${BLUE}Commands:${NC}"
  echo "  start       Start the NextGen trading system"
  echo "  stop        Stop the NextGen trading system"
  echo "  status      Check the status of all components"
  echo "  restart     Restart the NextGen trading system"
  echo ""
  echo -e "${BLUE}Options:${NC}"
  echo "  --help, -h  Show this help message"
  exit 1
}

# Function to check if a process is running
check_process() {
  local pid=$1
  local name=$2
  
  if [ -z "$pid" ]; then
    echo -e "${RED}$name is not running${NC}"
    return 1
  fi
  
  if ps -p $pid > /dev/null; then
    echo -e "${GREEN}$name is running (PID: $pid)${NC}"
    return 0
  else
    echo -e "${RED}$name is not running (stale PID: $pid)${NC}"
    return 1
  fi
}

# Function to start the Redis server
start_redis() {
  echo -e "${YELLOW}Starting Redis server...${NC}"
  
  # Check if Redis server is already running
  if [ -f "$PID_DIR/redis_server.pid" ]; then
    pid=$(cat "$PID_DIR/redis_server.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${GREEN}Redis server is already running (PID: $pid)${NC}"
      return 0
    else
      echo -e "${YELLOW}Removing stale PID file...${NC}"
      rm "$PID_DIR/redis_server.pid"
    fi
  fi
  
  # Start Redis server
  cd "$SCRIPT_DIR"
  python3 -m local_redis.start_redis_server > "$LOGS_DIR/redis_server.log" 2>&1 &
  REDIS_PID=$!
  
  echo $REDIS_PID > "$PID_DIR/redis_server.pid"
  echo -e "${GREEN}Redis server started with PID $REDIS_PID${NC}"
  
  # Wait a moment and check if the process is still running
  sleep 2
  if ! ps -p $REDIS_PID > /dev/null; then
    echo -e "${RED}Redis server failed to start. Check $LOGS_DIR/redis_server.log for details.${NC}"
    return 1
  fi
  
  return 0
}

# Function to start the monitoring system
start_monitoring() {
  echo -e "${YELLOW}Starting monitoring system...${NC}"
  
  # Check if monitoring system is already running
  if [ -f "$PID_DIR/monitoring.pid" ]; then
    pid=$(cat "$PID_DIR/monitoring.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${GREEN}Monitoring system is already running (PID: $pid)${NC}"
      return 0
    else
      echo -e "${YELLOW}Removing stale PID file...${NC}"
      rm "$PID_DIR/monitoring.pid"
    fi
  fi
  
  # Start monitoring system
  cd "$SCRIPT_DIR"
  "$SCRIPT_DIR/monitoring/start_monitoring_system.sh" > "$LOGS_DIR/monitoring.log" 2>&1 &
  MONITORING_PID=$!
  
  echo $MONITORING_PID > "$PID_DIR/monitoring.pid"
  echo -e "${GREEN}Monitoring system started with PID $MONITORING_PID${NC}"
  
  # Also record the dashboard PID if available
  if [ -f "$SCRIPT_DIR/monitoring/dashboard.pid" ]; then
    cp "$SCRIPT_DIR/monitoring/dashboard.pid" "$PID_DIR/dashboard.pid"
  fi
  
  # Also record the metrics PID if available
  if [ -f "$SCRIPT_DIR/monitoring/metrics.pid" ]; then
    cp "$SCRIPT_DIR/monitoring/metrics.pid" "$PID_DIR/metrics.pid"
  fi
  
  # Wait a moment and check if the process is still running
  sleep 2
  if ! ps -p $MONITORING_PID > /dev/null; then
    echo -e "${YELLOW}Main monitoring process exited, but child processes may be running${NC}"
  fi
  
  # Start the log server specifically
  start_log_server
  
  return 0
}

# Function to start the log server
start_log_server() {
  echo -e "${YELLOW}Starting log server...${NC}"
  
  # Check if log server is already running
  if [ -f "$PID_DIR/log_server.pid" ]; then
    pid=$(cat "$PID_DIR/log_server.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${GREEN}Log server is already running (PID: $pid)${NC}"
      return 0
    else
      echo -e "${YELLOW}Removing stale PID file...${NC}"
      rm "$PID_DIR/log_server.pid"
    fi
  fi
  
  # Start log server
  cd "$SCRIPT_DIR"
  python3 -m monitoring.start_log_server > "$LOGS_DIR/log_server.log" 2>&1 &
  LOG_SERVER_PID=$!
  
  echo $LOG_SERVER_PID > "$PID_DIR/log_server.pid"
  echo -e "${GREEN}Log server started with PID $LOG_SERVER_PID${NC}"
  
  # Wait a moment and check if the process is still running
  sleep 2
  if ! ps -p $LOG_SERVER_PID > /dev/null; then
    echo -e "${RED}Log server failed to start. Check $LOGS_DIR/log_server.log for details.${NC}"
    return 1
  fi
  
  return 0
}

# Function to start the MCP servers
start_mcp_servers() {
  echo -e "${YELLOW}Starting MCP servers...${NC}"
  
  # List of MCP servers to start
  # These will be started using the Python module imports
  local mcp_servers=(
    "data_mcp.polygon_news_mcp"
    "data_mcp.polygon_rest_mcp"
    "data_mcp.polygon_ws_mcp" 
    "data_mcp.reddit_mcp"
    "data_mcp.unusual_whales_mcp"
    "data_mcp.yahoo_finance_mcp"
    "data_mcp.yahoo_news_mcp"
    "db_mcp.redis_mcp"
    "document_analysis_mcp.document_analysis_mcp"
    "financial_data_mcp.financial_data_mcp"
    "financial_text_mcp.financial_text_mcp"
    "risk_analysis_mcp.risk_analysis_mcp"
    "time_series_mcp.time_series_mcp"
    "trading_mcp.trading_mcp"
    "vector_store_mcp.vector_store_mcp"
  )
  
  for mcp_server in "${mcp_servers[@]}"; do
    # Extract the server name for logging and PID file
    server_name=$(basename "$mcp_server" | cut -d. -f1)
    echo -e "${YELLOW}Starting $server_name...${NC}"
    
    # Check if this MCP server is already running
    if [ -f "$PID_DIR/${server_name}.pid" ]; then
      pid=$(cat "$PID_DIR/${server_name}.pid")
      if ps -p $pid > /dev/null; then
        echo -e "${GREEN}$server_name is already running (PID: $pid)${NC}"
        continue
      else
        echo -e "${YELLOW}Removing stale PID file for $server_name...${NC}"
        rm "$PID_DIR/${server_name}.pid"
      fi
    fi
    
    # Start the MCP server as a module
    cd "$SCRIPT_DIR"
    python3 -c "
from mcp_tools.$mcp_server import main

if __name__ == '__main__':
    main()
" > "$LOGS_DIR/${server_name}.log" 2>&1 &
    
    MCP_PID=$!
    echo $MCP_PID > "$PID_DIR/${server_name}.pid"
    echo -e "${GREEN}$server_name started with PID $MCP_PID${NC}"
    
    # Wait a moment to avoid overwhelming the system
    sleep 0.5
  done
  
  echo -e "${GREEN}All MCP servers started${NC}"
  return 0
}

# Function to start the NextGen models
start_models() {
  echo -e "${YELLOW}Starting NextGen models...${NC}"
  
  # List of model modules to start
  local models=(
    "nextgen_models.autogen_orchestrator.autogen_model"
    "nextgen_models.nextgen_context_model.context_model"
    "nextgen_models.nextgen_decision.decision_model"
    "nextgen_models.nextgen_fundamental_analysis.fundamental_analysis_model"
    "nextgen_models.nextgen_market_analysis.market_analysis_model"
    "nextgen_models.nextgen_risk_assessment.risk_assessment_model"
    "nextgen_models.nextgen_select.select_model"
    "nextgen_models.nextgen_sentiment_analysis.sentiment_analysis_model"
    "nextgen_models.nextgen_trader.trade_model"
  )
  
  for model in "${models[@]}"; do
    # Extract the model name for logging and PID file
    model_name=$(basename "$model" | cut -d. -f1)
    echo -e "${YELLOW}Starting $model_name...${NC}"
    
    # Check if this model is already running
    if [ -f "$PID_DIR/${model_name}.pid" ]; then
      pid=$(cat "$PID_DIR/${model_name}.pid")
      if ps -p $pid > /dev/null; then
        echo -e "${GREEN}$model_name is already running (PID: $pid)${NC}"
        continue
      else
        echo -e "${YELLOW}Removing stale PID file for $model_name...${NC}"
        rm "$PID_DIR/${model_name}.pid"
      fi
    fi
    
    # Start the model as a module
    cd "$SCRIPT_DIR"
    python3 -c "
import $model
if hasattr($model, 'main'):
    $model.main()
elif '__main__' in dir($model):
    print('Module has __main__ but no main() function')
else:
    print('Module initialized')
" > "$LOGS_DIR/${model_name}.log" 2>&1 &
    
    MODEL_PID=$!
    echo $MODEL_PID > "$PID_DIR/${model_name}.pid"
    echo -e "${GREEN}$model_name started with PID $MODEL_PID${NC}"
    
    # Wait a moment to avoid overwhelming the system
    sleep 0.5
  done
  
  echo -e "${GREEN}All NextGen models started${NC}"
  return 0
}

# Function to stop the Redis server
stop_redis() {
  echo -e "${YELLOW}Stopping Redis server...${NC}"
  
  if [ -f "$PID_DIR/redis_server.pid" ]; then
    pid=$(cat "$PID_DIR/redis_server.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${YELLOW}Sending SIGTERM to Redis server (PID: $pid)${NC}"
      kill -15 $pid
      
      # Wait for process to terminate
      for i in {1..10}; do
        if ! ps -p $pid > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # If process is still running, force kill
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Redis server not responding to SIGTERM, sending SIGKILL...${NC}"
        kill -9 $pid
      fi
      
      echo -e "${GREEN}Redis server stopped${NC}"
    else
      echo -e "${RED}Redis server was not running (stale PID file)${NC}"
    fi
    
    rm "$PID_DIR/redis_server.pid"
  else
    echo -e "${RED}Redis server was not running (no PID file)${NC}"
  fi
}

# Function to stop the monitoring system
stop_monitoring() {
  echo -e "${YELLOW}Stopping monitoring system...${NC}"
  
  # First stop the dashboard server if it's running
  if [ -f "$PID_DIR/dashboard.pid" ]; then
    pid=$(cat "$PID_DIR/dashboard.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${YELLOW}Sending SIGTERM to dashboard server (PID: $pid)${NC}"
      kill -15 $pid
      
      # Wait for process to terminate
      for i in {1..5}; do
        if ! ps -p $pid > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # If process is still running, force kill
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Dashboard server not responding to SIGTERM, sending SIGKILL...${NC}"
        kill -9 $pid
      fi
      
      echo -e "${GREEN}Dashboard server stopped${NC}"
    fi
    
    rm "$PID_DIR/dashboard.pid"
  fi
  
  # Stop the metrics collector if it's running
  if [ -f "$PID_DIR/metrics.pid" ]; then
    pid=$(cat "$PID_DIR/metrics.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${YELLOW}Sending SIGTERM to metrics collector (PID: $pid)${NC}"
      kill -15 $pid
      
      # Wait for process to terminate
      for i in {1..5}; do
        if ! ps -p $pid > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # If process is still running, force kill
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Metrics collector not responding to SIGTERM, sending SIGKILL...${NC}"
        kill -9 $pid
      fi
      
      echo -e "${GREEN}Metrics collector stopped${NC}"
    fi
    
    rm "$PID_DIR/metrics.pid"
  fi
  
  # Stop the log server
  stop_log_server
  
  # Stop the main monitoring process
  if [ -f "$PID_DIR/monitoring.pid" ]; then
    pid=$(cat "$PID_DIR/monitoring.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${YELLOW}Sending SIGTERM to monitoring system (PID: $pid)${NC}"
      kill -15 $pid
      
      # Wait for process to terminate
      for i in {1..5}; do
        if ! ps -p $pid > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # If process is still running, force kill
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Monitoring system not responding to SIGTERM, sending SIGKILL...${NC}"
        kill -9 $pid
      fi
      
      echo -e "${GREEN}Monitoring system stopped${NC}"
    else
      echo -e "${RED}Monitoring system was not running (stale PID file)${NC}"
    fi
    
    rm "$PID_DIR/monitoring.pid"
  else
    echo -e "${RED}Monitoring system was not running (no PID file)${NC}"
  fi
}

# Function to stop the log server
stop_log_server() {
  echo -e "${YELLOW}Stopping log server...${NC}"
  
  if [ -f "$PID_DIR/log_server.pid" ]; then
    pid=$(cat "$PID_DIR/log_server.pid")
    if ps -p $pid > /dev/null; then
      echo -e "${YELLOW}Sending SIGTERM to log server (PID: $pid)${NC}"
      kill -15 $pid
      
      # Wait for process to terminate
      for i in {1..5}; do
        if ! ps -p $pid > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # If process is still running, force kill
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Log server not responding to SIGTERM, sending SIGKILL...${NC}"
        kill -9 $pid
      fi
      
      echo -e "${GREEN}Log server stopped${NC}"
    else
      echo -e "${RED}Log server was not running (stale PID file)${NC}"
    fi
    
    rm "$PID_DIR/log_server.pid"
  else
    echo -e "${RED}Log server was not running (no PID file)${NC}"
  fi
}

# Function to stop the MCP servers
stop_mcp_servers() {
  echo -e "${YELLOW}Stopping MCP servers...${NC}"
  
  # Find all MCP server PID files
  for pid_file in "$PID_DIR"/*_mcp.pid; do
    if [ -f "$pid_file" ]; then
      server_name=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Sending SIGTERM to $server_name (PID: $pid)${NC}"
        kill -15 $pid
        
        # Wait for process to terminate
        for i in {1..5}; do
          if ! ps -p $pid > /dev/null; then
            break
          fi
          sleep 1
        done
        
        # If process is still running, force kill
        if ps -p $pid > /dev/null; then
          echo -e "${YELLOW}$server_name not responding to SIGTERM, sending SIGKILL...${NC}"
          kill -9 $pid
        fi
        
        echo -e "${GREEN}$server_name stopped${NC}"
      else
        echo -e "${RED}$server_name was not running (stale PID file)${NC}"
      fi
      
      rm "$pid_file"
    fi
  done
  
  echo -e "${GREEN}All MCP servers stopped${NC}"
}

# Function to stop the NextGen models
stop_models() {
  echo -e "${YELLOW}Stopping NextGen models...${NC}"
  
  # List of model names to stop (in reverse order from starting)
  local models=(
    "trade_model"
    "sentiment_analysis_model"
    "select_model"
    "risk_assessment_model"
    "market_analysis_model"
    "fundamental_analysis_model"
    "decision_model"
    "context_model"
    "autogen_model"
  )
  
  for model_name in "${models[@]}"; do
    if [ -f "$PID_DIR/${model_name}.pid" ]; then
      pid=$(cat "$PID_DIR/${model_name}.pid")
      
      if ps -p $pid > /dev/null; then
        echo -e "${YELLOW}Sending SIGTERM to $model_name (PID: $pid)${NC}"
        kill -15 $pid
        
        # Wait for process to terminate
        for i in {1..5}; do
          if ! ps -p $pid > /dev/null; then
            break
          fi
          sleep 1
        done
        
        # If process is still running, force kill
        if ps -p $pid > /dev/null; then
          echo -e "${YELLOW}$model_name not responding to SIGTERM, sending SIGKILL...${NC}"
          kill -9 $pid
        fi
        
        echo -e "${GREEN}$model_name stopped${NC}"
      else
        echo -e "${RED}$model_name was not running (stale PID file)${NC}"
      fi
      
      rm "$PID_DIR/${model_name}.pid"
    else
      echo -e "${RED}$model_name was not running (no PID file)${NC}"
    fi
  done
  
  echo -e "${GREEN}All NextGen models stopped${NC}"
}

# Function to show status of all components
status() {
  echo -e "${BLUE}NextGen System Status${NC}"
  echo -e "${BLUE}====================${NC}"
  
  echo -e "${BLUE}Redis server:${NC}"
  if [ -f "$PID_DIR/redis_server.pid" ]; then
    pid=$(cat "$PID_DIR/redis_server.pid")
    check_process "$pid" "Redis server"
  else
    echo -e "${RED}Redis server is not running${NC}"
  fi
  
  echo -e "\n${BLUE}Monitoring system:${NC}"
  if [ -f "$PID_DIR/monitoring.pid" ]; then
    pid=$(cat "$PID_DIR/monitoring.pid")
    check_process "$pid" "Monitoring system"
  else
    echo -e "${RED}Monitoring system is not running${NC}"
  fi
  
  if [ -f "$PID_DIR/dashboard.pid" ]; then
    pid=$(cat "$PID_DIR/dashboard.pid")
    check_process "$pid" "Dashboard server"
  else
    echo -e "${RED}Dashboard server is not running${NC}"
  fi
  
  if [ -f "$PID_DIR/metrics.pid" ]; then
    pid=$(cat "$PID_DIR/metrics.pid")
    check_process "$pid" "Metrics collector"
  else
    echo -e "${RED}Metrics collector is not running${NC}"
  fi
  
  if [ -f "$PID_DIR/log_server.pid" ]; then
    pid=$(cat "$PID_DIR/log_server.pid")
    check_process "$pid" "Log server"
  else
    echo -e "${RED}Log server is not running${NC}"
  fi
  
  echo -e "\n${BLUE}MCP servers:${NC}"
  mcp_count=0
  for pid_file in "$PID_DIR"/*_mcp.pid; do
    if [ -f "$pid_file" ]; then
      server_name=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      check_process "$pid" "$server_name"
      ((mcp_count++))
    fi
  done
  
  if [ $mcp_count -eq 0 ]; then
    echo -e "${RED}No MCP servers are running${NC}"
  fi
  
  echo -e "\n${BLUE}NextGen models:${NC}"
  model_count=0
  for pid_file in "$PID_DIR"/*_model.pid; do
    if [ -f "$pid_file" ]; then
      model_name=$(basename "$pid_file" .pid)
      pid=$(cat "$pid_file")
      check_process "$pid" "$model_name"
      ((model_count++))
    fi
  done
  
  if [ $model_count -eq 0 ]; then
    echo -e "${RED}No NextGen models are running${NC}"
  fi
}

# Function to start the entire system
start_system() {
  echo -e "${GREEN}Starting NextGen trading system...${NC}"
  
  # Start Redis first as other components depend on it
  start_redis
  
  # Start monitoring system
  start_monitoring
  
  # Start MCP servers
  start_mcp_servers
  
  # Start NextGen models
  start_models
  
  echo -e "${GREEN}NextGen trading system started${NC}"
  echo -e "${GREEN}Use '$0 status' to check the status of all components${NC}"
}

# Function to stop the entire system
stop_system() {
  echo -e "${YELLOW}Stopping NextGen trading system...${NC}"
  
  # Stop in reverse order of starting
  # First stop models
  stop_models
  
  # Stop MCP servers
  stop_mcp_servers
  
  # Stop monitoring system
  stop_monitoring
  
  # Stop Redis server last
  stop_redis
  
  echo -e "${GREEN}NextGen trading system stopped${NC}"
}

# Function to restart the entire system
restart_system() {
  echo -e "${YELLOW}Restarting NextGen trading system...${NC}"
  stop_system
  sleep 2
  start_system
}

# Parse command-line arguments
if [ $# -eq 0 ]; then
  usage
fi

case "$1" in
  start)
    start_system
    ;;
  stop)
    stop_system
    ;;
  status)
    status
    ;;
  restart)
    restart_system
    ;;
  --help|-h)
    usage
    ;;
  *)
    echo -e "${RED}Unknown command: $1${NC}"
    usage
    ;;
esac

exit 0