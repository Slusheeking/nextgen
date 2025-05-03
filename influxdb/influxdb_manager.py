import os
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from influxdb import InfluxDBClient as InfluxDBClientV1
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

class InfluxDBManager:
    """
    A manager class for InfluxDB functionality.
    Provides an interface for interacting with both InfluxDB 1.x and 2.x databases, 
    including writing points, querying data, and managing databases.
    """
    
    def __init__(self, url=None, username=None, password=None, token=None, org=None, 
                 bucket=None, verify_ssl=None, api_version=None, timeout=None):
        """
        Initialize the InfluxDB manager.
        
        Args:
            url (str, optional): InfluxDB server URL including protocol, host and port.
                               Defaults to env var INFLUXDB_URL or 'http://localhost:8086'.
            username (str, optional): Username for authentication. Defaults to env var INFLUXDB_USER_NAME or None.
            password (str, optional): Password for authentication. Defaults to env var INFLUXDB_PASSWORD or None.
            token (str, optional): Auth token for InfluxDB 2.x. Defaults to env var INFLUXDB_TOKEN or None.
            org (str, optional): Organization for InfluxDB 2.x. Defaults to env var INFLUXDB_ORG or None.
            bucket (str, optional): Bucket for InfluxDB 2.x. Defaults to env var INFLUXDB_BUCKET or 'metrics'.
            verify_ssl (bool, optional): Whether to verify SSL certificates. Defaults to env var or True.
            api_version (str, optional): API version ('v1' or 'v2'). Defaults to env var or 'v2'.
            timeout (int, optional): Connection timeout in seconds. Defaults to env var or 10.
        """
        # Load environment variables
        load_dotenv()
        
        # Get environment details
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        
        # Parse URL
        self.url = url or os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        parsed_url = urllib.parse.urlparse(self.url)
        self.host = parsed_url.hostname or 'localhost'
        self.port = parsed_url.port or 8086
        self.ssl = parsed_url.scheme == 'https'
        
        # Authentication settings
        self.username = username or os.getenv('INFLUXDB_USER_NAME', None)
        self.password = password or os.getenv('INFLUXDB_PASSWORD', None)
        
        # InfluxDB 2.x specific settings
        self.token = token or os.getenv('INFLUXDB_TOKEN', None)
        self.org = org or os.getenv('INFLUXDB_ORG', 'nextgen')
        self.bucket = bucket or os.getenv('INFLUXDB_BUCKET', 'market_data')
        
        # Legacy database for v1 API
        self.database = self.bucket
        
        # Handle SSL verification setting
        env_verify_ssl = os.getenv('INFLUXDB_VERIFY_SSL', 'True')
        self.verify_ssl = verify_ssl if verify_ssl is not None else (env_verify_ssl.lower() == 'true')
        
        # API version defaults to v2
        self.api_version = api_version or os.getenv('INFLUXDB_API_VERSION', 'v2')
        
        # Timeout
        env_timeout = os.getenv('INFLUXDB_TIMEOUT', '10')
        self.timeout = int(timeout or env_timeout)
        
        # Initialize client based on API version
        self.client = None
        self.write_api = None
        self.query_api = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate InfluxDB client based on API version."""
        try:
            if self.api_version.lower() == 'v1':
                # For v1, use URL or host/port
                if self.url and not (self.host and self.port):
                    self.client = InfluxDBClientV1(
                        url=self.url,
                        username=self.username,
                        password=self.password,
                        database=self.database,
                        ssl=self.ssl,
                        verify_ssl=self.verify_ssl,
                        timeout=self.timeout
                    )
                else:
                    self.client = InfluxDBClientV1(
                        host=self.host,
                        port=self.port,
                        username=self.username,
                        password=self.password,
                        database=self.database,
                        ssl=self.ssl,
                        verify_ssl=self.verify_ssl,
                        timeout=self.timeout
                    )
                # Create database if it doesn't exist
                if self.database:
                    self._ensure_database_exists()
            else:  # v2
                # For v2, use URL from the input or env var directly
                self.client = InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org,
                    timeout=self.timeout * 1000,  # ms
                    verify_ssl=self.verify_ssl
                )
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
        except Exception as e:
            print(f"Error initializing InfluxDB client: {e}")
            self.client = None
    
    def _ensure_database_exists(self):
        """Create the database if it doesn't exist (v1 only)."""
        if self.api_version.lower() == 'v1' and self.client:
            try:
                databases = self.client.get_list_database()
                if not any(db['name'] == self.database for db in databases):
                    self.client.create_database(self.database)
            except Exception as e:
                print(f"Error creating database '{self.database}': {e}")
    
    def reconfigure(self, url=None, username=None, password=None, token=None, 
                   org=None, bucket=None, verify_ssl=None, api_version=None, timeout=None):
        """
        Reconfigure the InfluxDB client with new settings.
        
        Args:
            url (str, optional): New InfluxDB server URL including protocol, host and port.
            username (str, optional): New username for authentication.
            password (str, optional): New password for authentication.
            token (str, optional): New auth token for InfluxDB 2.x.
            org (str, optional): New organization for InfluxDB 2.x.
            bucket (str, optional): New bucket for InfluxDB 2.x.
            verify_ssl (bool, optional): New verify SSL certificates setting.
            api_version (str, optional): New API version ('v1' or 'v2').
            timeout (int, optional): New connection timeout in seconds.
        """
        # Update settings if provided
        if url:
            self.url = url
            parsed_url = urllib.parse.urlparse(self.url)
            self.host = parsed_url.hostname or 'localhost'
            self.port = parsed_url.port or 8086
            self.ssl = parsed_url.scheme == 'https'
            
        if username is not None:
            self.username = username
        if password is not None:
            self.password = password
        if token is not None:
            self.token = token
        if org is not None:
            self.org = org
        if bucket:
            self.bucket = bucket
            # Update database for v1 compatibility
            self.database = bucket
        if verify_ssl is not None:
            self.verify_ssl = verify_ssl
        if api_version:
            self.api_version = api_version
        if timeout:
            self.timeout = int(timeout)
        
        # Close old client if exists
        self.close()
        
        # Initialize new client with updated settings
        self._initialize_client()
    
    def ping(self) -> bool:
        """
        Ping the InfluxDB server to check connectivity.
        
        Returns:
            bool: True if server responds, False otherwise.
        """
        try:
            if self.api_version.lower() == 'v1':
                return self.client.ping()
            else:  # v2
                self.client.ping()
                return True
        except Exception as e:
            print(f"InfluxDB ping failed: {e}")
            return False
    
    def write_point(self, measurement: str, fields: Dict[str, Any], 
                    tags: Optional[Dict[str, str]] = None, 
                    timestamp: Optional[Union[int, datetime]] = None,
                    database: Optional[str] = None,
                    bucket: Optional[str] = None,
                    retention_policy: Optional[str] = None) -> bool:
        """
        Write a single data point to InfluxDB.
        
        Args:
            measurement (str): Measurement name.
            fields (dict): Field key-value pairs.
            tags (dict, optional): Tag key-value pairs.
            timestamp (int or datetime, optional): Custom timestamp for the data point.
            database (str, optional): Database to write to (v1 only). Defaults to self.database.
            bucket (str, optional): Bucket to write to (v2 only). Defaults to self.bucket.
            retention_policy (str, optional): Retention policy (v1 only).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return False
        
        try:
            if self.api_version.lower() == 'v1':
                point = {
                    "measurement": measurement,
                    "fields": fields
                }
                
                if tags:
                    point["tags"] = tags
                
                if timestamp:
                    if isinstance(timestamp, datetime):
                        point["time"] = timestamp.isoformat()
                    else:
                        point["time"] = timestamp
                
                db_to_use = database or self.database
                return self.client.write_points(
                    [point], 
                    database=db_to_use,
                    retention_policy=retention_policy,
                    time_precision='ms'
                )
            else:  # v2
                point = Point(measurement)
                
                # Add fields
                for field_name, field_value in fields.items():
                    if isinstance(field_value, bool):
                        point.field(field_name, field_value)
                    elif isinstance(field_value, int):
                        point.field(field_name, field_value)
                    elif isinstance(field_value, float):
                        point.field(field_name, field_value)
                    else:
                        point.field(field_name, str(field_value))
                
                # Add tags
                if tags:
                    for tag_name, tag_value in tags.items():
                        point.tag(tag_name, tag_value)
                
                # Add timestamp
                if timestamp:
                    if isinstance(timestamp, datetime):
                        point.time(timestamp, WritePrecision.MS)
                    else:
                        point.time(timestamp, WritePrecision.MS)
                
                bucket_to_use = bucket or self.bucket
                self.write_api.write(bucket=bucket_to_use, org=self.org, record=point)
                return True
        except Exception as e:
            print(f"Error writing point to InfluxDB: {e}")
            return False
    
    def write_points(self, points: List[Dict[str, Any]], 
                     database: Optional[str] = None,
                     bucket: Optional[str] = None,
                     retention_policy: Optional[str] = None) -> bool:
        """
        Write multiple data points to InfluxDB.
        
        Args:
            points (list): List of point dictionaries (for v1) or Point objects (for v2).
            database (str, optional): Database to write to (v1 only). Defaults to self.database.
            bucket (str, optional): Bucket to write to (v2 only). Defaults to self.bucket.
            retention_policy (str, optional): Retention policy (v1 only).
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return False
        
        try:
            if self.api_version.lower() == 'v1':
                db_to_use = database or self.database
                return self.client.write_points(
                    points, 
                    database=db_to_use,
                    retention_policy=retention_policy,
                    time_precision='ms'
                )
            else:  # v2
                bucket_to_use = bucket or self.bucket
                self.write_api.write(bucket=bucket_to_use, org=self.org, record=points)
                return True
        except Exception as e:
            print(f"Error writing points to InfluxDB: {e}")
            return False
    
    def query(self, query: str, database: Optional[str] = None, 
              bucket: Optional[str] = None, org: Optional[str] = None,
              epoch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Execute a query against the InfluxDB database.
        
        Args:
            query (str): Query string (InfluxQL for v1, Flux for v2).
            database (str, optional): Database to query (v1 only). Defaults to self.database.
            bucket (str, optional): Bucket to query (v2 only). Defaults to self.bucket.
            org (str, optional): Organization (v2 only). Defaults to self.org.
            epoch (str, optional): Precision for timestamp results (v1 only).
        
        Returns:
            list: Query results as a list of dictionaries.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return []
        
        try:
            if self.api_version.lower() == 'v1':
                db_to_use = database or self.database
                params = {}
                if epoch:
                    params['epoch'] = epoch
                
                result = self.client.query(query, database=db_to_use, params=params)
                return list(result.get_points())
            else:  # v2
                org_to_use = org or self.org
                # For v2, Flux queries typically include the bucket in the query string
                # so bucket parameter might not be needed, but we'll add it for clarity
                result = self.query_api.query(query=query, org=org_to_use)
                
                # Convert result to a list of dictionaries
                output = []
                for table in result:
                    for record in table.records:
                        row = {
                            # Add the field values
                            **record.values,
                            # Add time if present
                            'time': record.get_time() 
                        }
                        output.append(row)
                
                return output
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def create_database(self, database_name: str) -> bool:
        """
        Create a new database (v1) or bucket (v2).
        
        Args:
            database_name (str): Name of the database/bucket to create.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return False
        
        try:
            if self.api_version.lower() == 'v1':
                self.client.create_database(database_name)
                return True
            else:  # v2
                # In v2, we need to use the bucket API
                buckets_api = self.client.buckets_api()
                organization_api = self.client.organizations_api()
                
                # Get the organization ID
                orgs = organization_api.find_organizations(org=self.org)
                if not orgs:
                    print(f"Organization '{self.org}' not found")
                    return False
                
                org_id = orgs[0].id
                
                # Create a bucket with infinite retention by default
                buckets_api.create_bucket(bucket_name=database_name, org_id=org_id)
                return True
        except Exception as e:
            print(f"Error creating database '{database_name}': {e}")
            return False
    
    def drop_database(self, database_name: str) -> bool:
        """
        Drop a database (v1) or bucket (v2).
        
        Args:
            database_name (str): Name of the database/bucket to drop.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return False
        
        try:
            if self.api_version.lower() == 'v1':
                self.client.drop_database(database_name)
                return True
            else:  # v2
                # In v2, we need to use the bucket API
                buckets_api = self.client.buckets_api()
                
                # Find the bucket
                buckets = buckets_api.find_buckets(name=database_name)
                if not buckets:
                    print(f"Bucket '{database_name}' not found")
                    return False
                
                # Delete the bucket
                buckets_api.delete_bucket(buckets[0])
                return True
        except Exception as e:
            print(f"Error dropping database '{database_name}': {e}")
            return False
    
    def list_databases(self) -> List[str]:
        """
        List all databases (v1) or buckets (v2).
        
        Returns:
            list: List of database/bucket names.
        """
        if not self.client:
            print("InfluxDB client not initialized")
            return []
        
        try:
            if self.api_version.lower() == 'v1':
                databases = self.client.get_list_database()
                return [db['name'] for db in databases]
            else:  # v2
                buckets_api = self.client.buckets_api()
                buckets = buckets_api.find_buckets()
                return [bucket.name for bucket in buckets]
        except Exception as e:
            print(f"Error listing databases: {e}")
            return []
    
    def create_retention_policy(self, name: str, duration: str, 
                               replication: int = 1, database: Optional[str] = None, 
                               default: bool = False) -> bool:
        """
        Create a retention policy (v1 only).
        
        Args:
            name (str): Name of the retention policy.
            duration (str): Duration specification (e.g., '1h', '7d', 'INF').
            replication (int, optional): Replication factor. Defaults to 1.
            database (str, optional): Database to create the policy on. Defaults to self.database.
            default (bool, optional): Whether to set as default policy. Defaults to False.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.client or self.api_version.lower() != 'v1':
            print("This operation is only supported in InfluxDB v1")
            return False
        
        try:
            db_to_use = database or self.database
            self.client.create_retention_policy(
                name=name,
                duration=duration,
                replication=replication,
                database=db_to_use,
                default=default
            )
            return True
        except Exception as e:
            print(f"Error creating retention policy '{name}': {e}")
            return False
    
    def close(self):
        """Close the InfluxDB connection."""
        try:
            if self.client:
                if self.api_version.lower() == 'v2':
                    if self.write_api:
                        self.write_api.close()
                    self.client.close()
                else:  # v1
                    self.client.close()
        except Exception as e:
            print(f"Error closing InfluxDB connection: {e}")


# Example usage
if __name__ == "__main__":
    # Create an InfluxDB manager using environment variables
    influxdb_manager = InfluxDBManager()
    
    # Alternatively, you can explicitly provide parameters:
    # influxdb_manager = InfluxDBManager(
    #     url=os.getenv('INFLUXDB_URL'),
    #     token=os.getenv('INFLUXDB_TOKEN'),
    #     org=os.getenv('INFLUXDB_ORG'),
    #     bucket=os.getenv('INFLUXDB_BUCKET')
    # )
    
    # Check connection
    if influxdb_manager.ping():
        print("Connected to InfluxDB successfully")
    else:
        print("Failed to connect to InfluxDB")
        exit(1)
    
    # Write a single point
    success = influxdb_manager.write_point(
        measurement="system",
        fields={"cpu": 0.5, "memory": 60.4},
        tags={"host": "server-1", "region": "us-west"}
    )
    print(f"Write point: {'Success' if success else 'Failed'}")
    
    # Write multiple points
    points = []
    for i in range(3):
        if influxdb_manager.api_version.lower() == 'v1':
            points.append({
                "measurement": "system",
                "tags": {"host": f"server-{i}", "region": "us-west"},
                "fields": {"cpu": 0.5 + i*0.1, "memory": 60.0 + i*5.0},
                "time": int(time.time() * 1000)  # millisecond precision
            })
        else:  # v2
            point = Point("system")
            point.tag("host", f"server-{i}")
            point.tag("region", "us-west")
            point.field("cpu", 0.5 + i*0.1)
            point.field("memory", 60.0 + i*5.0)
            point.time(time.time_ns())  # nanosecond precision
            points.append(point)
    
    success = influxdb_manager.write_points(points)
    print(f"Write points: {'Success' if success else 'Failed'}")
    
    # Query data
    if influxdb_manager.api_version.lower() == 'v1':
        # InfluxQL query for v1
        results = influxdb_manager.query(
            "SELECT * FROM system WHERE time > now() - 1h"
        )
    else:
        # Flux query for v2
        results = influxdb_manager.query(
            f'from(bucket:"{influxdb_manager.bucket}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "system")'
        )
    
    print(f"Query results: {results}")
    
    # Close connection
    influxdb_manager.close()
