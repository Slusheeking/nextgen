import os
import json
import redis
from dotenv import load_dotenv

class RedisManager:
    """
    A manager class for Redis functionality.
    Provides an interface for interacting with Redis, supporting various Redis data types
    and operations.
    """
    
    def __init__(self, host=None, port=None, db=None, password=None, 
                 username=None, ssl=None, socket_timeout=None):
        """
        Initialize the Redis manager.
        
        Args:
            host (str, optional): Redis server host. Defaults to env var REDIS_HOST or 'localhost'.
            port (int, optional): Redis server port. Defaults to env var REDIS_PORT or 6379.
            db (int, optional): Redis DB number. Defaults to env var REDIS_DB or 0.
            password (str, optional): Redis password. Defaults to env var REDIS_PASSWORD or None.
            username (str, optional): Redis username. Defaults to env var REDIS_USERNAME or None.
            ssl (bool, optional): Whether to use SSL. Defaults to env var REDIS_SSL or False.
            socket_timeout (int, optional): Socket timeout in seconds. Defaults to env var or None.
        """
        # Load environment variables
        load_dotenv()
        
        # Get environment details
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.service_version = os.getenv('SERVICE_VERSION', '1.0.0')
        
        # Set configuration from parameters or environment variables
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = int(port or os.getenv('REDIS_PORT', '6379'))
        self.db = int(db or os.getenv('REDIS_DB', '0'))
        self.password = password or os.getenv('REDIS_PASSWORD', None)
        self.username = username or os.getenv('REDIS_USERNAME', None)
        
        # Handle SSL setting from string env var if needed
        env_ssl = os.getenv('REDIS_SSL', 'False')
        if ssl is None:
            self.ssl = env_ssl.lower() == 'true'
        else:
            self.ssl = ssl
        
        # Socket timeout
        env_timeout = os.getenv('REDIS_SOCKET_TIMEOUT', None)
        if socket_timeout is None and env_timeout is not None:
            self.socket_timeout = float(env_timeout)
        else:
            self.socket_timeout = socket_timeout
        
        # Initialize the Redis client
        self.client = self._create_client()
    
    def _create_client(self):
        """
        Create a Redis client with current settings.
        
        Returns:
            redis.Redis: Configured Redis client.
        """
        # Prepare connection arguments
        connection_args = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'ssl': self.ssl
        }
        
        # Add optional arguments if provided
        if self.password:
            connection_args['password'] = self.password
        if self.username:
            connection_args['username'] = self.username
        if self.socket_timeout is not None:
            connection_args['socket_timeout'] = self.socket_timeout
        
        # Create and return the client
        return redis.Redis(**connection_args)
    
    def reconfigure(self, host=None, port=None, db=None, password=None, 
                    username=None, ssl=None, socket_timeout=None):
        """
        Reconfigure the Redis client with new settings.
        
        Args:
            host (str, optional): New Redis server host.
            port (int, optional): New Redis server port.
            db (int, optional): New Redis DB number.
            password (str, optional): New Redis password.
            username (str, optional): New Redis username.
            ssl (bool, optional): New SSL setting.
            socket_timeout (int, optional): New socket timeout in seconds.
        """
        # Update settings if provided
        if host:
            self.host = host
        if port:
            self.port = int(port)
        if db is not None:
            self.db = int(db)
        if password:
            self.password = password
        if username:
            self.username = username
        if ssl is not None:
            self.ssl = ssl
        if socket_timeout is not None:
            self.socket_timeout = socket_timeout
        
        # Close old client if exists
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass
        
        # Create new client with updated settings
        self.client = self._create_client()
    
    def ping(self):
        """
        Ping the Redis server to check connectivity.
        
        Returns:
            bool: True if server responds, False otherwise.
        """
        try:
            return self.client.ping()
        except Exception as e:
            print(f"Redis ping failed: {e}")
            return False
    
    def get(self, key):
        """
        Get a value by key.
        
        Args:
            key (str): The key to retrieve.
        
        Returns:
            The value if found, None otherwise.
        """
        try:
            value = self.client.get(key)
            if value is not None:
                # Try to decode as JSON, fallback to string
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            print(f"Redis get error for key '{key}': {e}")
            return None
    
    def set(self, key, value, expiry=None):
        """
        Set a key-value pair, with optional expiry.
        
        Args:
            key (str): The key to set.
            value: The value to set, can be string or JSON-serializable object.
            expiry (int, optional): Expiry time in seconds.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Convert non-string/bytes values to JSON
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
            
            if expiry:
                return self.client.setex(key, expiry, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            print(f"Redis set error for key '{key}': {e}")
            return False
    
    def delete(self, *keys):
        """
        Delete one or more keys.
        
        Args:
            *keys: Keys to delete.
        
        Returns:
            int: Number of keys deleted.
        """
        try:
            return self.client.delete(*keys)
        except Exception as e:
            print(f"Redis delete error: {e}")
            return 0
    
    def exists(self, key):
        """
        Check if a key exists.
        
        Args:
            key (str): Key to check.
        
        Returns:
            bool: True if key exists, False otherwise.
        """
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            print(f"Redis exists error for key '{key}': {e}")
            return False
    
    def expire(self, key, seconds):
        """
        Set expiration time for a key.
        
        Args:
            key (str): The key to set expiration for.
            seconds (int): Expiration time in seconds.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return self.client.expire(key, seconds)
        except Exception as e:
            print(f"Redis expire error for key '{key}': {e}")
            return False
    
    def keys(self, pattern="*"):
        """
        Find all keys matching a pattern.
        
        Args:
            pattern (str): Pattern to match (default: "*").
        
        Returns:
            list: List of matching keys.
        """
        try:
            keys = self.client.keys(pattern)
            return [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]
        except Exception as e:
            print(f"Redis keys error for pattern '{pattern}': {e}")
            return []
    
    # Hash operations
    def hset(self, name, key, value):
        """
        Set a hash field to value.
        
        Args:
            name (str): Name of the hash.
            key (str): Field name.
            value: Field value.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Convert non-string/bytes values to JSON
            if not isinstance(value, (str, bytes, int, float)):
                value = json.dumps(value)
            
            return bool(self.client.hset(name, key, value))
        except Exception as e:
            print(f"Redis hset error for name '{name}', key '{key}': {e}")
            return False
    
    def hget(self, name, key):
        """
        Get a hash field value.
        
        Args:
            name (str): Name of the hash.
            key (str): Field name.
        
        Returns:
            The field value if found, None otherwise.
        """
        try:
            value = self.client.hget(name, key)
            if value is not None:
                # Try to decode as JSON, fallback to string
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            print(f"Redis hget error for name '{name}', key '{key}': {e}")
            return None
    
    def hgetall(self, name):
        """
        Get all fields and values in a hash.
        
        Args:
            name (str): Name of the hash.
        
        Returns:
            dict: Dictionary of field-value pairs.
        """
        try:
            result = {}
            hash_data = self.client.hgetall(name)
            
            for k, v in hash_data.items():
                # Decode keys
                key = k.decode('utf-8') if isinstance(k, bytes) else k
                
                # Try to decode values as JSON, fallback to string
                try:
                    value = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    value = v.decode('utf-8') if isinstance(v, bytes) else v
                
                result[key] = value
                
            return result
        except Exception as e:
            print(f"Redis hgetall error for name '{name}': {e}")
            return {}
    
    def hdel(self, name, *keys):
        """
        Delete one or more hash fields.
        
        Args:
            name (str): Name of the hash.
            *keys: Field names to delete.
        
        Returns:
            int: Number of fields deleted.
        """
        try:
            return self.client.hdel(name, *keys)
        except Exception as e:
            print(f"Redis hdel error for name '{name}': {e}")
            return 0
    
    # List operations
    def lpush(self, name, *values):
        """
        Push values to the left of a list.
        
        Args:
            name (str): Name of the list.
            *values: Values to push.
        
        Returns:
            int: Length of the list after push.
        """
        try:
            # Convert complex values to JSON
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            
            return self.client.lpush(name, *serialized_values)
        except Exception as e:
            print(f"Redis lpush error for name '{name}': {e}")
            return 0
    
    def rpush(self, name, *values):
        """
        Push values to the right of a list.
        
        Args:
            name (str): Name of the list.
            *values: Values to push.
        
        Returns:
            int: Length of the list after push.
        """
        try:
            # Convert complex values to JSON
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            
            return self.client.rpush(name, *serialized_values)
        except Exception as e:
            print(f"Redis rpush error for name '{name}': {e}")
            return 0
    
    def lrange(self, name, start, end):
        """
        Get a range of elements from a list.
        
        Args:
            name (str): Name of the list.
            start (int): Start index.
            end (int): End index.
        
        Returns:
            list: Range of elements.
        """
        try:
            values = self.client.lrange(name, start, end)
            result = []
            
            for v in values:
                # Try to decode as JSON, fallback to string
                try:
                    result.append(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.append(v.decode('utf-8') if isinstance(v, bytes) else v)
            
            return result
        except Exception as e:
            print(f"Redis lrange error for name '{name}': {e}")
            return []
    
    def ltrim(self, name, start, end):
        """
        Trim a list to a specified range.
        
        Args:
            name (str): Name of the list.
            start (int): Start index.
            end (int): End index.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.client.ltrim(name, start, end)
            return True
        except Exception as e:
            print(f"Redis ltrim error for name '{name}': {e}")
            return False
    
    # Set operations
    def sadd(self, name, *values):
        """
        Add values to a set.
        
        Args:
            name (str): Name of the set.
            *values: Values to add.
        
        Returns:
            int: Number of values added.
        """
        try:
            # Convert complex values to JSON
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            
            return self.client.sadd(name, *serialized_values)
        except Exception as e:
            print(f"Redis sadd error for name '{name}': {e}")
            return 0
    
    def smembers(self, name):
        """
        Get all members of a set.
        
        Args:
            name (str): Name of the set.
        
        Returns:
            set: Set of members.
        """
        try:
            values = self.client.smembers(name)
            result = set()
            
            for v in values:
                # Try to decode as JSON, fallback to string
                try:
                    result.add(json.loads(v))
                except (json.JSONDecodeError, TypeError):
                    result.add(v.decode('utf-8') if isinstance(v, bytes) else v)
            
            return result
        except Exception as e:
            print(f"Redis smembers error for name '{name}': {e}")
            return set()
    
    def srem(self, name, *values):
        """
        Remove values from a set.
        
        Args:
            name (str): Name of the set.
            *values: Values to remove.
        
        Returns:
            int: Number of values removed.
        """
        try:
            # Convert complex values to JSON for consistent comparison
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, bytes, int, float)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)
            
            return self.client.srem(name, *serialized_values)
        except Exception as e:
            print(f"Redis srem error for name '{name}': {e}")
            return 0
    
    # Sorted set operations
    def zadd(self, name, mapping):
        """
        Add to a sorted set with scores.
        
        Args:
            name (str): Name of the sorted set.
            mapping (dict): Dict of {value: score} to add.
        
        Returns:
            int: Number of values added.
        """
        try:
            # Convert complex values to JSON
            serialized_mapping = {}
            for k, v in mapping.items():
                if not isinstance(k, (str, bytes, int, float)):
                    serialized_mapping[json.dumps(k)] = v
                else:
                    serialized_mapping[k] = v
            
            return self.client.zadd(name, serialized_mapping)
        except Exception as e:
            print(f"Redis zadd error for name '{name}': {e}")
            return 0
    
    def zrange(self, name, start, end, withscores=False):
        """
        Get a range of elements from a sorted set.
        
        Args:
            name (str): Name of the sorted set.
            start (int): Start index.
            end (int): End index.
            withscores (bool): Return scores with elements.
        
        Returns:
            list: Range of elements.
        """
        try:
            values = self.client.zrange(name, start, end, withscores=withscores)
            
            if not withscores:
                result = []
                for v in values:
                    # Try to decode as JSON, fallback to string
                    try:
                        result.append(json.loads(v))
                    except (json.JSONDecodeError, TypeError):
                        result.append(v.decode('utf-8') if isinstance(v, bytes) else v)
                return result
            else:
                result = []
                for i in range(0, len(values), 2):
                    v = values[i]
                    score = values[i+1]
                    
                    # Try to decode as JSON, fallback to string
                    try:
                        value = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        value = v.decode('utf-8') if isinstance(v, bytes) else v
                    
                    result.append((value, score))
                return result
        except Exception as e:
            print(f"Redis zrange error for name '{name}': {e}")
            return []
    
    # Pub/Sub operations
    def publish(self, channel, message):
        """
        Publish a message to a channel.
        
        Args:
            channel (str): Channel to publish to.
            message: Message to publish.
        
        Returns:
            int: Number of clients that received the message.
        """
        try:
            # Convert complex values to JSON
            if not isinstance(message, (str, bytes, int, float)):
                message = json.dumps(message)
            
            return self.client.publish(channel, message)
        except Exception as e:
            print(f"Redis publish error for channel '{channel}': {e}")
            return 0
    
    def subscribe(self, callback, *channels):
        """
        Subscribe to channels and process messages with a callback.
        
        Args:
            callback (function): Function to call for each message.
            *channels: Channels to subscribe to.
        """
        pubsub = self.client.pubsub()
        pubsub.subscribe(*channels)
        
        # Process messages
        for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                
                # Try to decode as JSON, fallback to string
                try:
                    if isinstance(data, bytes):
                        data = json.loads(data)
                    else:
                        data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    data = data.decode('utf-8') if isinstance(data, bytes) else data
                
                # Call the callback with channel and data
                callback(message['channel'], data)
    
    def close(self):
        """Close the Redis connection."""
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            print(f"Redis close error: {e}")


# Example usage
if __name__ == "__main__":
    # Create a Redis manager
    redis_manager = RedisManager()
    
    # Check connection
    if redis_manager.ping():
        print("Connected to Redis successfully")
    else:
        print("Failed to connect to Redis")
        exit(1)
    
    # String operations
    redis_manager.set("test:string", "Hello, Redis!")
    redis_manager.set("test:json", {"name": "Redis", "type": "NoSQL"})
    
    # Set an expiring key
    redis_manager.set("test:expiring", "I will disappear", expiry=60)
    
    # Get values
    print(f"String value: {redis_manager.get('test:string')}")
    print(f"JSON value: {redis_manager.get('test:json')}")
    
    # Hash operations
    redis_manager.hset("test:hash", "field1", "value1")
    redis_manager.hset("test:hash", "field2", {"nested": "value"})
    
    print(f"Hash field: {redis_manager.hget('test:hash', 'field1')}")
    print(f"All hash: {redis_manager.hgetall('test:hash')}")
    
    # List operations
    redis_manager.lpush("test:list", "first", "second")
    redis_manager.rpush("test:list", "last", {"complex": "value"})
    
    print(f"List range: {redis_manager.lrange('test:list', 0, -1)}")
    
    # Set operations
    redis_manager.sadd("test:set", "value1", "value2", {"set": "value"})
    print(f"Set members: {redis_manager.smembers('test:set')}")
    
    # Cleanup
    redis_manager.delete("test:string", "test:json", "test:expiring", 
                        "test:hash", "test:list", "test:set")
    
    # Close connection
    redis_manager.close()
