import redis
from src.logger import LOG_TYPES, logger
import dotenv
import os

dotenv.load_dotenv() 

def test_redis_connection() -> bool:
    r = redis.Redis(host=os.getenv('REDIS_HOST'), port=int(os.getenv('REDIS_PORT')), decode_responses=True)

    try:
        if(r.ping()):
            logger('Successfully connected to Redis server', LOG_TYPES.SUCCESS)
            return True
        return False
    except Exception as e:
        logger(f"Redis connection failed: {e}", LOG_TYPES.ERROR)
        return False
    finally:
        r.close()

def get_redis_client() -> redis.Redis:
    """Initialize and return a Redis client."""

    try:
        redis_client = redis.Redis(
            host='localhost', 
            port=int(os.getenv('REDIS_PORT')), 
            max_connections=100,
            decode_responses=True
        )

        # Test Redis connection
        redis_client.ping()
        logger("Redis client initialized successfully", LOG_TYPES.SUCCESS)
        return redis_client
    
    except Exception as e:
        logger(f"Redis connection failed: {e}", LOG_TYPES.ERROR)
        redis_client = None
        return redis_client
    
def store_user_message_to_redis(redis_client: redis.Redis, user_uuid: str, message: str, message_type: str = "user_message") -> bool:
    """Store user message to Redis as simple string."""
    if not redis_client:
        logger("Redis client not available, skipping message storage", LOG_TYPES.WARNING)
        return False  
        
    try:
        # Store as simple string format: "user_message: <msg>"
        message_string = f"{message_type}: {message}"
        
        # Store in Redis using user_uuid as key
        redis_key = f"user_chat:{user_uuid}"
        redis_client.rpush(redis_key, message_string)
        
        # Set expiration for the key (e.g., 24 hours)
        redis_client.expire(redis_key, 86400)
        
        logger(f"Stored '{message_string}' to Redis for user {user_uuid}", LOG_TYPES.SUCCESS)
        return True
        
    except Exception as e:
        logger(f"Failed to store message to Redis: {e}", LOG_TYPES.ERROR)
        return False

def get_user_chat_context(redis_client: redis.Redis,user_uuid: str, limit: int = 10) -> list:
    """Get user chat context from Redis."""
    if not redis_client:
        return []
        
    try:
        redis_key = f"user_chat:{user_uuid}"
        messages = redis_client.lrange(redis_key, 0, limit - 1)
        return messages  
    except Exception as e:
        logger(f"Failed to get chat context from Redis: {e}", LOG_TYPES.ERROR)
        return []

def remove_user_session(redis_client: redis.Redis, user_uuid: str) -> bool:
    """Remove user session from Redis when user leaves."""
    if not redis_client:
        return False
        
    try:
        redis_key = f"user_chat:{user_uuid}"
        result = redis_client.delete(redis_key)
        if result:
            logger(f"Removed session for user {user_uuid}", LOG_TYPES.SUCCESS)
        return bool(result)
    except Exception as e:
        logger(f"Failed to remove user session: {e}", LOG_TYPES.ERROR)
        return False

if __name__ == "__main__":
    print(test_redis_connection())



