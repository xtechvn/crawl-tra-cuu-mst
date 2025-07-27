import aio_pika
import json
import logging
import asyncio
from config import config
from redis import get_redis
import time

logger = logging.getLogger(__name__)

class TaxCrawlResultConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.result_queue = None
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
            
            # Khai báo result queue
            self.result_queue = await self.channel.declare_queue(
                config.RABBITMQ_RESULT_QUEUE,
                durable=True
            )
            
            # Bind result queue với exchange
            await self.result_queue.bind(
                self.exchange,
                routing_key="tax.result"
            )
            
            logger.info("✅ RabbitMQ Result Consumer connected successfully")
            logger.info(f"📋 Result Queue: {config.RABBITMQ_RESULT_QUEUE}")
            logger.info(f"📋 Exchange: {config.RABBITMQ_EXCHANGE}")
            logger.info(f"📋 Routing Key: tax.result")
            
        except Exception as e:
            logger.error(f"❌ RabbitMQ Result Consumer connection failed: {e}")
            raise

    async def process_result(self, message: aio_pika.IncomingMessage):
        """Xử lý result message"""
        async with message.process():
            try:
                # Parse result message
                body = json.loads(message.body.decode())
                request_id = body.get("request_id")
                tax_no = body.get("tax_no")
                status = body.get("status")
                data = body.get("data")
                crawl_time = body.get("crawl_time_seconds")
                
                logger.info(f"📥 Received result: request_id={request_id}, tax_no={tax_no}, status={status}")
                
                # Lưu kết quả vào Redis với key là request_id
                redis = await get_redis()
                result_key = f"result:{request_id}"
                
                result_data = {
                    "request_id": request_id,
                    "tax_no": tax_no,
                    "status": status,
                    "data": data,
                    "crawl_time_seconds": crawl_time,
                    "timestamp": time.time(),
                    "completed": True
                }
                
                # Lưu với TTL 1 giờ
                await redis.set(result_key, json.dumps(result_data), ex=3600)
                
                logger.info(f"💾 Saved result to Redis: {result_key}")
                
            except Exception as e:
                logger.error(f"❌ Error processing result: {e}")

    async def start_consuming(self):
        """Bắt đầu consume result messages"""
        if not self.connection:
            await self.connect()
        
        logger.info(f"🎯 Starting result consumer for queue: {config.RABBITMQ_RESULT_QUEUE}")
        
        # Bắt đầu consume
        await self.result_queue.consume(self.process_result)
        
        try:
            # Giữ consumer chạy
            await asyncio.Future()  # Chạy vô hạn
        except KeyboardInterrupt:
            logger.info("🛑 Result Consumer stopped by user")
        finally:
            await self.close()

    async def close(self):
        """Đóng kết nối"""
        if self.connection:
            await self.connection.close()
            logger.info("🔌 RabbitMQ Result Consumer connection closed")

# Result consumer instance
result_consumer = TaxCrawlResultConsumer() 