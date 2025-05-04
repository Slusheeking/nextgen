# Local Vector Database Server

This package provides a local vector database service using ChromaDB that integrates with the NextGen system. It's designed to work with the VectorDBMCP to provide persistent storage and retrieval of vector embeddings.

## Features

- Persistent storage of vector embeddings and metadata
- Integration with system monitoring (Prometheus and Loki)
- Automatic service management through systemd
- Connection interface for VectorDBMCP
- Support for multiple collections
- Fault tolerance and automatic recovery

## Installation

To install and configure the local vector database service:

```bash
# Run the setup script with sudo
sudo ./setup_vectordb_server.sh
```

This will:
1. Install required dependencies (chromadb, sentence-transformers)
2. Create necessary directories
3. Set up the systemd service
4. Configure monitoring integration
5. Start the service

## Configuration

The vector database can be configured through environment variables or the .env file:

| Environment Variable | Description | Default Value |
|---------------------|-------------|---------------|
| VECTORDB_PATH | Path to store vector database files | ./vector_db_storage |
| VECTORDB_HOST | Host for HTTP server mode | localhost |
| VECTORDB_PORT | Port for HTTP server mode | 8000 |
| VECTORDB_USE_HTTP_SERVER | Use HTTP server mode instead of local | False |
| PROMETHEUS_METRICS_PORT | Port for Prometheus metrics | 8011 |

## Usage with VectorDBMCP

To use the local vector database with VectorDBMCP, configure VectorDBMCP to use ChromaDB with the appropriate path:

```python
from mcp_tools.analysis_mcp.vector_db_mcp import VectorDBMCP

# Configure VectorDBMCP to use the local vector database
config = {
    "db_type": "chroma",  # Use ChromaDB
    "db_path": "./vector_db_storage",  # Path to the storage directory
    "default_collection": "my_collection"
}

# Initialize VectorDBMCP
vector_db_mcp = VectorDBMCP(config)

# Use VectorDBMCP as normal
vector_db_mcp.add_embeddings(...)
vector_db_mcp.query_similar(...)
```

## Service Management

The vector database runs as a systemd service that automatically starts on boot and restarts on failure:

```bash
# Check service status
sudo systemctl status vectordb

# Restart the service
sudo systemctl restart vectordb

# Stop the service
sudo systemctl stop vectordb

# View logs
sudo journalctl -u vectordb
```

## Monitoring

The service exports metrics to Prometheus and sends logs to Loki for monitoring:

- Collection count
- Embedding counts per collection
- Operation counts and durations
- Error rates

## Architecture

The vector database server uses ChromaDB's PersistentClient by default, which provides local persistence of embeddings and metadata. It can also be configured to use HttpClient mode for connecting to a remote ChromaDB server if needed.

```
+------------------+    +-----------------+    +------------------+
|    VectorDBMCP   | -> | VectorDBServer  | -> |    ChromaDB      |
+------------------+    +-----------------+    +------------------+
                                                       |
                                                       v
                                               +----------------+
                                               | vector_db_storage |
                                               +----------------+