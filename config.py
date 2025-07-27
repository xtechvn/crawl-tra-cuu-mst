import os
from typing import Optional

class Config:
    """Configuration class for the application"""
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    # REDIS
    ADDRESS_REDIS: str = os.getenv("ADDRESS_REDIS", "redis://103.163.216.42:6666/4")    
    # Crawler Configuration
    CRAWLER_BASE_URL: str = "https://tracuunnt.gdt.gov.vn/"
    CRAWLER_TIMEOUT: int = int(os.getenv("CRAWLER_TIMEOUT", "30"))
    CRAWLER_RETRY_COUNT: int = int(os.getenv("CRAWLER_RETRY_COUNT", "3"))
    CRAWLER_DELAY: float = float(os.getenv("CRAWLER_DELAY", "2.0"))
    
    # Chrome Driver Configuration
    CHROME_HEADLESS: bool = os.getenv("CHROME_HEADLESS", "false").lower() == "true"
    CHROME_WINDOW_SIZE: str = os.getenv("CHROME_WINDOW_SIZE", "1920,1080")
    CHROME_DISABLE_IMAGES: bool = os.getenv("CHROME_DISABLE_IMAGES", "false").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/tax_crawler.log"
    
    # Task Configuration
    MAX_CONCURRENT_TASKS: int = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    TASK_TIMEOUT: int = int(os.getenv("TASK_TIMEOUT", "300"))  # 5 minutes
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds    
  
    # RabbitMQ Configuration
    RABBITMQ_URL: str = os.getenv(
        "RABBITMQ_URL",
        "amqp://crawl:123456@103.163.216.41:5672/crawler"
    )
    RABBITMQ_QUEUE_NAME: str = os.getenv("RABBITMQ_QUEUE_NAME", "tax_crawl_queue")
    RABBITMQ_RESULT_QUEUE: str = os.getenv("RABBITMQ_RESULT_QUEUE", "tax_crawl_result")
    RABBITMQ_EXCHANGE: str = os.getenv("RABBITMQ_EXCHANGE", "tax_crawl_exchange")
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

# Create config instance
config = Config() 