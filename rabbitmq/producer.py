import aio_pika
import json
import logging
from config import config
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class TaxCrawlProducer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue = None
        self.exchange = None

    async def connect(self):
        """Kết nối đến RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(config.RABBITMQ_URL)
            self.channel = await self.connection.channel()
            
            # Khai báo exchange
            self.exchange = await self.channel.declare_exchange(
                config.RABBITMQ_EXCHANGE,
                aio_pika.ExchangeType.TOPIC
            )
            
            # Khai báo queue
            self.queue = await self.channel.declare_queue(
                config.RABBITMQ_QUEUE_NAME,
                durable=True
            )
            
            # Bind queue với exchange
            await self.queue.bind(
                self.exchange,
                routing_key="tax.crawl"
            )
            
            logger.info("✅ RabbitMQ Producer connected successfully")
            logger.info(f"📋 Queue: {config.RABBITMQ_QUEUE_NAME}")
            logger.info(f"📋 Exchange: {config.RABBITMQ_EXCHANGE}")
            logger.info(f"📋 Routing Key: tax.crawl")
            
        except Exception as e:
            logger.error(f"❌ RabbitMQ Producer connection failed: {e}")
            raise

    async def publish_crawl_task(self, tax_no: str, request_id: str = None) -> str:
        """Gửi task crawl vào queue"""
        if not self.connection:
            await self.connect()
        
        # Tạo message
        message = {
            "tax_no": tax_no,
            "request_id": request_id or f"req_{tax_no}_{int(time.time())}",
            "timestamp": time.time()
        }
        
        # Gửi message
        await self.exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"request_id": request_id}
            ),
            routing_key="tax.crawl"
        )
        
        logger.info(f"📤 Published crawl task for tax_no: {tax_no}, request_id: {message['request_id']}")
        return message['request_id']

    async def close(self):
        """Đóng kết nối"""
        if self.connection:
            await self.connection.close()
            logger.info("🔌 RabbitMQ Producer connection closed")

# Global producer instance
producer = TaxCrawlProducer() 