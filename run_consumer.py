#!/usr/bin/env python3
"""
Script chạy RabbitMQ Consumer Worker
"""
import asyncio
import logging
from rabbitmq.consumer import consumer
from utils.logger import setup_logger
from config import config

# Setup logging
logger = setup_logger(level=config.LOG_LEVEL)

async def main():
    """Main function để chạy consumer"""
    logger.info("🚀 Starting Tax Crawl Consumer Worker...")
    
    try:
        # Kết nối và bắt đầu consume
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("🛑 Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Consumer error: {e}")
    finally:
        await consumer.close()

if __name__ == "__main__":
    asyncio.run(main()) 