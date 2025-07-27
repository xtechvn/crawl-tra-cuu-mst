import aioredis
from config import config

# Global redis client
_redis_client = None

async def get_redis():
    global _redis_client
    if _redis_client is None:        
        redis_url = config.ADDRESS_REDIS
        _redis_client = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    return _redis_client