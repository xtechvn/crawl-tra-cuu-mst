#!/usr/bin/env python3
"""
Test script để kết nối RabbitMQ và tạo queue
"""
import asyncio
import aio_pika
from config import config

async def test_rabbitmq_connection():
    """Test kết nối RabbitMQ"""
    try:
        print(f"🔗 Connecting to RabbitMQ: {config.RABBITMQ_URL}")
        
        # Kết nối
        connection = await aio_pika.connect_robust(config.RABBITMQ_URL)
        channel = await connection.channel()
        
        print("✅ Connected to RabbitMQ successfully!")
        
        # Khai báo exchange
        exchange = await channel.declare_exchange(
            config.RABBITMQ_EXCHANGE,
            aio_pika.ExchangeType.TOPIC
        )
        print(f"📋 Exchange created: {config.RABBITMQ_EXCHANGE}")
        
        # Khai báo input queue
        input_queue = await channel.declare_queue(
            config.RABBITMQ_QUEUE_NAME,
            durable=True
        )
        print(f"📋 Input Queue created: {config.RABBITMQ_QUEUE_NAME}")
        
        # Khai báo result queue
        result_queue = await channel.declare_queue(
            config.RABBITMQ_RESULT_QUEUE,
            durable=True
        )
        print(f"📋 Result Queue created: {config.RABBITMQ_RESULT_QUEUE}")
        
        # Bind queues với exchange
        await input_queue.bind(exchange, routing_key="tax.crawl")
        await result_queue.bind(exchange, routing_key="tax.result")
        print("🔗 Queues bound to exchange")
        
        # Gửi test message
        test_message = {
            "tax_no": "test_123",
            "request_id": "test_req_123",
            "timestamp": 1234567890
        }
        
        await exchange.publish(
            aio_pika.Message(
                body=str(test_message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key="tax.crawl"
        )
        print("📤 Test message sent to queue")
        
        # Đóng kết nối
        await connection.close()
        print("🔌 Connection closed")
        
        print("\n🎉 RabbitMQ setup completed successfully!")
        print("📋 Now check RabbitMQ Management UI to see the queues:")
        print(f"   - Input Queue: {config.RABBITMQ_QUEUE_NAME}")
        print(f"   - Result Queue: {config.RABBITMQ_RESULT_QUEUE}")
        print(f"   - Exchange: {config.RABBITMQ_EXCHANGE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rabbitmq_connection()) 