#!/usr/bin/env python3
"""
Script chạy RabbitMQ Result Consumer
"""
import asyncio
import logging
from rabbitmq.result_consumer import result_consumer
from utils.logger import setup_logger
from config import config

# Setup logging
logger = setup_logger(level=config.LOG_LEVEL)

async def main():
    """Main function để chạy result consumer"""
    logger.info("🚀 Starting Tax Crawl Result Consumer...")
    
    try:
        # Kết nối và bắt đầu consume
        await result_consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("🛑 Result Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Result Consumer error: {e}")
    finally:
        await result_consumer.close()

if __name__ == "__main__":
    asyncio.run(main()) 