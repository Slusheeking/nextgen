"""
Log API Server

This module provides a FastAPI server that serves log files to the front end.
It reads logs from the log directory and provides endpoints to:
- Get a list of available log files
- Get the content of a specific log file
- Stream real-time log updates

Usage:
    python log_api.py
"""

import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
import asyncio
from datetime import datetime
import json

# Create FastAPI app
app = FastAPI(
    title="NextGen Logs API",
    description="API for accessing NextGen application logs",
    version="1.0.0",
)

# Add CORS middleware to allow requests from the front end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the log directory path
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Create a master log file if it doesn't exist
MASTER_LOG = os.path.join(LOG_DIR, "master.log")
if not os.path.exists(MASTER_LOG):
    with open(MASTER_LOG, "w") as f:
        f.write(f"--- Master log created at {datetime.now().isoformat()} ---\n")


class LogEntry(BaseModel):
    """Model for a log entry"""
    timestamp: str
    level: str
    service: str
    message: str
    tags: Optional[Dict[str, str]] = None


class LogFile(BaseModel):
    """Model for log file information"""
    name: str
    path: str
    size: int
    last_modified: str


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "NextGen Logs API",
        "version": "1.0.0",
        "endpoints": [
            "/logs - Get list of log files",
            "/logs/{filename} - Get content of a specific log file",
            "/logs/{filename}/stream - Stream updates to a log file",
            "/logs/master - Get content of the master log file",
            "/logs/master/stream - Stream updates to the master log file",
        ]
    }


@app.get("/logs", response_model=List[LogFile])
async def get_logs():
    """Get a list of available log files"""
    log_files = []
    
    try:
        for filename in os.listdir(LOG_DIR):
            if filename.endswith(".log"):
                file_path = os.path.join(LOG_DIR, filename)
                stats = os.stat(file_path)
                log_files.append(
                    LogFile(
                        name=filename,
                        path=file_path,
                        size=stats.st_size,
                        last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    )
                )
        return log_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log directory: {str(e)}")


@app.get("/logs/{filename}")
async def get_log_content(
    filename: str, 
    lines: Optional[int] = Query(None, description="Number of lines to return from the end of the file"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR, etc.)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
):
    """Get the content of a specific log file"""
    file_path = os.path.join(LOG_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Log file {filename} not found")
    
    try:
        with open(file_path, "r") as f:
            content = f.readlines()
        
        # Apply filters if specified
        if level or service:
            filtered_content = []
            for line in content:
                if level and level.upper() not in line.upper():
                    continue
                if service and service not in line:
                    continue
                filtered_content.append(line)
            content = filtered_content
        
        # Return only the specified number of lines from the end
        if lines:
            content = content[-lines:]
        
        return {"filename": filename, "content": "".join(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")


async def log_stream(filename):
    """Generator function for streaming log updates"""
    file_path = os.path.join(LOG_DIR, filename)
    
    if not os.path.exists(file_path):
        yield f"data: {json.dumps({'error': f'Log file {filename} not found'})}\n\n"
        return
    
    # Start at the end of the file
    with open(file_path, "r") as f:
        f.seek(0, os.SEEK_END)
        position = f.tell()
    
    while True:
        with open(file_path, "r") as f:
            f.seek(position)
            new_content = f.read()
            position = f.tell()
        
        if new_content:
            # Send each new line as a separate event
            for line in new_content.splitlines():
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
        
        await asyncio.sleep(1)  # Check for updates every second


@app.get("/logs/{filename}/stream")
async def stream_log(filename: str):
    """Stream updates to a log file using Server-Sent Events (SSE)"""
    return StreamingResponse(
        log_stream(filename),
        media_type="text/event-stream",
    )


@app.get("/logs/master")
async def get_master_log(
    lines: Optional[int] = Query(None, description="Number of lines to return from the end of the file"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR, etc.)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
):
    """Get the content of the master log file"""
    return await get_log_content("master.log", lines, level, service)


@app.get("/logs/master/stream")
async def stream_master_log():
    """Stream updates to the master log file"""
    return StreamingResponse(
        log_stream("master.log"),
        media_type="text/event-stream",
    )


@app.post("/logs")
async def add_log_entry(entry: LogEntry):
    """Add a log entry to the master log file"""
    try:
        with open(MASTER_LOG, "a") as f:
            tags_str = ""
            if entry.tags:
                tags_str = " " + " ".join([f"{k}={v}" for k, v in entry.tags.items()])
            
            log_line = f"{entry.timestamp} - {entry.service} - {entry.level} - {entry.message}{tags_str}\n"
            f.write(log_line)
        return {"status": "success", "message": "Log entry added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to log file: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "log_api:app",
        host="0.0.0.0",
        port=8011,
        reload=True,
    )