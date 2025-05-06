from chromadb.config import Settings, System # Import System
from chromadb.api.fastapi import FastAPI # Import the FastAPI app class
import uvicorn
import time
import socket
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_config():
    """Get configuration from environment variables with fallbacks"""
    host = os.getenv("CHROMADB_HOST", "localhost")
    
    # Get port from environment, fallback to a list of ports to try
    port_str = os.getenv("CHROMADB_PORT")
    if port_str:
        try:
            # If specific port is provided, only try that one
            return host, [int(port_str)]
        except ValueError:
            print(f"Invalid CHROMADB_PORT value: {port_str}. Using default ports.")
    
    # Default list of ports to try if no specific port is provided
    return host, [8005, 8006, 8007, 8008, 8009]

if __name__ == "__main__":
    # Get host and ports from environment
    host, ports_to_try = get_env_config()

    for port in ports_to_try:
        try:
            print(f"Attempting to start ChromaDB server on {host}:{port}...")
            # Initialize ChromaDB settings with the current port
            settings = Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=host,
                chroma_server_http_port=port,
                anonymized_telemetry=False # Disable telemetry
            )

            # Create a ChromaDB System instance
            system = System(settings)

            # Initialize the FastAPI app using the system
            # The FastAPI app itself is the entry point for uvicorn
            app = FastAPI(system) # Pass the system object

            # Run the server using uvicorn
            # We use a try-except block around uvicorn.run to catch port binding errors
            uvicorn.run(app, host=host, port=port)

            # If uvicorn.run succeeds, the server is running, break the loop
            print(f"ChromaDB server started successfully on {host}:{port}")
            break # Exit the loop if successful

        except socket.gaierror as e:
            print(f"Error resolving host '{host}' for port {port}: {e}")
            # This is a configuration error, likely not port related, so we might want to exit or handle differently
            # For now, let's just log and continue to the next port (though resolving localhost should always work)
            time.sleep(1) # Wait a bit before trying the next port
            continue
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is already in use. Trying next port...")
                time.sleep(1) # Wait a bit before trying the next port
                continue # Try the next port
            else:
                print(f"An unexpected OS error occurred on port {port}: {e}")
                time.sleep(1) # Wait a bit before trying the next port
                continue # Try the next port
        except Exception as e:
            print(f"An unexpected error occurred on port {port}: {e}")
            time.sleep(1) # Wait a bit before trying the next port
            continue # Try the next port
    else:
        print("Failed to start ChromaDB server on any of the specified ports.")
        # Optionally, exit or raise an error if no port was successful
        # sys.exit(1) # Uncomment to exit if no port works
