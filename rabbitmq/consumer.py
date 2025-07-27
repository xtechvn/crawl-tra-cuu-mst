import aio_pika
import json
import logging
import asyncio
from config import config
from crawler.tax_crawler import TaxCrawler
from redis import get_redis
import time

logger = logging.getLogger(__name__)

class TaxCrawlConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue = None
        self.exchange = None
        self.result_queue = None
        self.crawler = TaxCrawler()

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
            
            # Khai báo input queue
            self.queue = await self.channel.declare_queue(
                config.RABBITMQ_QUEUE_NAME,
                durable=True
            )
            
            # Khai báo result queue
            self.result_queue = await self.channel.declare_queue(
                config.RABBITMQ_RESULT_QUEUE,
                durable=True
            )
            
            # Bind queues với exchange
            await self.queue.bind(
                self.exchange,
                routing_key="tax.crawl"
            )
            
            await self.result_queue.bind(
                self.exchange,
                routing_key="tax.result"
            )
            
            logger.info("✅ RabbitMQ Consumer connected successfully")
            logger.info(f"📋 Input Queue: {config.RABBITMQ_QUEUE_NAME}")
            logger.info(f"📋 Result Queue: {config.RABBITMQ_RESULT_QUEUE}")
            logger.info(f"📋 Exchange: {config.RABBITMQ_EXCHANGE}")
            logger.info(f"📋 Routing Keys: tax.crawl -> tax.result")
            
        except Exception as e:
            logger.error(f"❌ RabbitMQ Consumer connection failed: {e}")
            raise

    async def process_message(self, message: aio_pika.IncomingMessage):
        """Xử lý message từ queue"""
        async with message.process():
            try:
                # Parse message
                body = json.loads(message.body.decode())
                tax_no = body.get("tax_no")
                request_id = body.get("request_id")
                
                logger.info(f"🔄 Processing crawl task: tax_no={tax_no}, request_id={request_id}")
                logger.info(f"📥 Raw message body: {message.body.decode()}")
                
                # Thực hiện crawl
                start_time = time.time()
                status, data = await self.crawler.search_by_tax_code(tax_no)
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Cache kết quả cho tất cả trường hợp (success, no data, error)
                redis = await get_redis()
                cache_key = f"{tax_no}"
                
                # Lưu data hoặc error message
                cache_data = data if status == 0 else {"status": status, "message": "Không tìm thấy dữ liệu" if status == 1 else str(data)}
                await redis.set(cache_key, json.dumps(cache_data), ex=43200)  # 12 tiếng
                logger.info(f"💾 Cached result for tax_no: {tax_no}, status: {status}")
                
                # Tạo result message
                result = {
                    "request_id": request_id,
                    "tax_no": tax_no,
                    "status": status,
                    "data": data if status == 0 else None,
                    "crawl_time_seconds": elapsed_time,
                    "timestamp": time.time()
                }
                
                # Gửi kết quả về result queue
                await self.exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(result).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        headers={"request_id": request_id}
                    ),
                    routing_key="tax.result"
                )
                
                logger.info(f"✅ Completed crawl task: tax_no={tax_no}, status={status}, time={elapsed_time:.2f}s")
                logger.info(f"📤 Sending result to result queue: {result}")
                
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                # Gửi error result
                error_result = {
                    "request_id": body.get("request_id") if 'body' in locals() else "unknown",
                    "tax_no": body.get("tax_no") if 'body' in locals() else "unknown",
                    "status": 2,
                    "error": str(e),
                    "timestamp": time.time()
                }
                
                await self.exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(error_result).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                    ),
                    routing_key="tax.result"
                )

    async def start_consuming(self):
        """Bắt đầu consume messages"""
        if not self.connection:
            await self.connect()
        
        logger.info(f"🎯 Starting consumer for queue: {config.RABBITMQ_QUEUE_NAME}")
        
        # Bắt đầu consume
        await self.queue.consume(self.process_message)
        
        try:
            # Giữ consumer chạy
            await asyncio.Future()  # Chạy vô hạn
        except KeyboardInterrupt:
            logger.info("🛑 Consumer stopped by user")
        finally:
            await self.close()

    async def close(self):
        """Đóng kết nối"""
        if self.connection:
            await self.connection.close()
            logger.info("🔌 RabbitMQ Consumer connection closed")

# Consumer instance
consumer = TaxCrawlConsumer() 