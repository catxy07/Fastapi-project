import json
import redis
from app.core.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL)

# Functions to get and set cached predictions
def get_cached_prediction(key: str):
    cached_data = redis_client.get(key)
    if cached_data:
        return json.loads(cached_data)
    return None

# Function to set cached predictions with an expiration time
def set_cached_prediction(key: str, value: dict, expire_seconds: int = 3600):
    redis_client.setex(key, expire_seconds, json.dumps(value))
