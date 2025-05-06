import chromadb
from chromadb.config import Settings
import uvicorn

if __name__ == "__main__":
    # Initialize ChromaDB client with HTTP server settings
    settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host="localhost",
        chroma_server_http_port=8000,
        anonymized_telemetry=False # Disable telemetry
    )

    # Create a ChromaDB server instance
    server = chromadb.Server(settings)

    # Run the server using uvicorn
    uvicorn.run(server.app(), host="localhost", port=8000)
