import os
import logging
import logging_loki
import time
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

class LokiManager:
    """
    A manager class for Loki logging functionality.
    Implements a wrapper around python-logging-loki to send logs to a Loki instance.
    """
    
    def __init__(self, service_name=None, loki_url=None, loki_username=None, loki_password=None):
        """
        Initialize the Loki logger manager.
        
        Args:
            service_name (str, optional): Name of the service for log identification.
            loki_url (str, optional): URL of the Loki server including protocol, host and port.
                                     Defaults to env var LOKI_URL or 'http://localhost:3100'.
            loki_username (str, optional): Username for Loki authentication. Defaults to env var.
            loki_password (str, optional): Password for Loki authentication. Defaults to env var.
        """
        # Load environment variables
        load_dotenv()
        
        # Set service name from parameters or environment variables
        environment = os.getenv('ENVIRONMENT', 'development')
        service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        default_name = f"nextgen-{environment}"
        self.service_name = service_name or os.getenv('SERVICE_NAME', default_name)
        
        # Parse Loki URL from parameters or environment variables
        self.loki_url = loki_url or os.getenv('LOKI_URL', 'http://localhost:3100')
        
        # Parse the URL to extract host and port
        parsed_url = urlparse(self.loki_url)
        self.loki_host = parsed_url.hostname or 'localhost'
        self.loki_port = parsed_url.port or 3100
        self.loki_scheme = parsed_url.scheme or 'http'
        
        # Set authentication credentials
        self.loki_username = loki_username or os.getenv('LOKI_USERNAME', '')
        self.loki_password = loki_password or os.getenv('LOKI_PASSWORD', '')
        
        # Add version as a default tag
        self.version = service_version
        
        # Configure the Loki handler
        self._configure_loki_handler()
        
    def _configure_loki_handler(self):
        """Configure the Loki handler with current settings."""
        # Create authentication if credentials provided
        auth = None
        if self.loki_username and self.loki_password:
            auth = (self.loki_username, self.loki_password)
        
        # Create the full URL
        push_url = f"{self.loki_scheme}://{self.loki_host}:{self.loki_port}/loki/api/v1/push"
        
        # Create the Loki handler
        loki_handler = logging_loki.LokiHandler(
            url=push_url,
            tags={
                "service": self.service_name,
                "version": self.version,
                "environment": os.getenv('ENVIRONMENT', 'development')
            },
            auth=auth,
            version="1"
        )
        
        # Configure the root logger with the Loki handler
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(loki_handler)
        
        # Add a formatter for structured logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        loki_handler.setFormatter(formatter)
    
    def info(self, message, **labels):
        """
        Log an info message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.INFO, message, **labels)
    
    def warning(self, message, **labels):
        """
        Log a warning message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.WARNING, message, **labels)
    
    def error(self, message, **labels):
        """
        Log an error message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.ERROR, message, **labels)
    
    def critical(self, message, **labels):
        """
        Log a critical message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.CRITICAL, message, **labels)
    
    def debug(self, message, **labels):
        """
        Log a debug message to Loki.
        
        Args:
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        self._log(logging.DEBUG, message, **labels)
    
    def _log(self, level, message, **labels):
        """
        Internal method to log messages with additional labels.
        
        Args:
            level (int): Logging level (e.g., logging.INFO).
            message (str): The message to log.
            **labels: Additional labels to attach to the log entry.
        """
        # Add timestamp and service labels
        extra_labels = {
            'timestamp': time.time(),
            'service': self.service_name
        }
        
        # Add any additional labels provided
        extra_labels.update(labels)
        
        # Log the message with the extra labels
        self.logger.log(level, message, extra=extra_labels)
    
    def reconfigure(self, loki_url=None, loki_username=None, 
                   loki_password=None, service_name=None):
        """
        Reconfigure the Loki logger with new settings.
        
        Args:
            loki_url (str, optional): New URL of the Loki server including protocol, host and port.
            loki_username (str, optional): New username for Loki authentication.
            loki_password (str, optional): New password for Loki authentication.
            service_name (str, optional): New service name.
        """
        if loki_url:
            self.loki_url = loki_url
            # Re-parse the URL
            parsed_url = urlparse(self.loki_url)
            self.loki_host = parsed_url.hostname or 'localhost'
            self.loki_port = parsed_url.port or 3100
            self.loki_scheme = parsed_url.scheme or 'http'
            
        if loki_username:
            self.loki_username = loki_username
        if loki_password:
            self.loki_password = loki_password
        if service_name:
            self.service_name = service_name
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Reconfigure with new settings
        self._configure_loki_handler()


# Example usage
if __name__ == "__main__":
    # Create a Loki logger instance
    loki_logger = LokiManager(service_name="nextgen-test")
    
    # Log some messages
    loki_logger.info("Test information message", component="test", action="initialization")
    loki_logger.warning("Test warning message", component="test", action="process")
    loki_logger.error("Test error message", component="test", action="validation", error_code=500)
    
    # Reconfigure the logger with different settings
    loki_logger.reconfigure(loki_url="loki-server", loki_port=3100)
    
    # Log with the new configuration
    loki_logger.info("Test after reconfiguration", component="test", action="reconfiguration")
