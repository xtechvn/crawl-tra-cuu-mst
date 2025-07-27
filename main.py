from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any
from datetime import datetime
import os
import time
from crawler.tax_crawler import TaxCrawler
from utils.logger import setup_logger
from config import config
from fastapi import Query
import json
from redis import get_redis
# from rabbitmq.producer import producer

# Setup logger
logger = setup_logger(level=config.LOG_LEVEL)

# Create FastAPI app
app = FastAPI(
    title="Tax Data Crawler API",
    description="API để crawl dữ liệu từ trang tracuunnt.gdt.gov.vn",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tax Data Crawler API",
        "version": "1.0.0",
        "endpoint": "/search - Tìm kiếm thông tin doanh nghiệp"
    }


# Khởi tạo Redis client (kết nối 1 lần, reuse)
# Redis client được quản lý trong redis/connect.py

#0  là thành công
#1  là không có dữ liệu
@app.get("/search_use_cache")
async def search_company_noqueue(tax_no: str):
    """
    Tìm kiếm thông tin doanh nghiệp theo mã số thuế - Không dùng queue.
    Chỉ kiểm tra cache, nếu không có thì gọi trực tiếp search_by_tax_code.
    """
    try:
        is_set_cache = False
        redis = await get_redis()
        cache_key = f"{tax_no}"
        cached_data = await redis.get(cache_key)
        print(f"kiểm tra trong cache{cached_data}")
        if cached_data:
            logger.info(f"Trả về dữ liệu từ cache cho mst: {tax_no}")
            result = json.loads(cached_data)
            data = [0, result]
            crawl_time = 0
            is_set_cache = True
        else:        
            # Nếu không có cache, gọi trực tiếp hàm search_by_tax_code
            logger.info(f"Không có cache, bắt đầu crawl trực tiếp cho mst: {tax_no}")
            start_time = time.time()
            crawler = TaxCrawler()
            data = await crawler.search_by_tax_code(tax_no)
            crawl_time = round(time.time() - start_time, 2)

            if data[0] == 0: # Crawl thành công
                if data[1].get("status") == 1: # KIểm tra mã số thuế này có dữ liệu không                
                    await redis.set(cache_key, json.dumps(data), ex=43200)  # 12 tiếng        
                    is_set_cache = True
                else:
                    data[0] = 1 # Khoong co du lieu theo mã số thuế này
                    is_set_cache = False

        return {
            "data": data[1],
            "crawl_time_seconds": crawl_time,
            "cached": is_set_cache,
            "status": data[0]
        }
    except Exception as e:
        logger.error(f"Error searching for tax code {tax_no} (noqueue): {str(e)}")
        return {                        
            "ex": {str(e)},
            "status": 1
        }        

# @app.get("/search-push-queue")
# async def search_company(tax_no: str):
#     """Tìm kiếm thông tin doanh nghiệp theo mã số thuế - Async với RabbitMQ"""
#     try:
#         # Kiểm tra cache trước
#         redis = await get_redis()
#         cache_key = f"{tax_no}"
#         cached_data = await redis.get(cache_key)
        
#         if cached_data:
#             logger.info(f"Trả về dữ liệu từ cache cho mst: {tax_no}")
#             result = json.loads(cached_data)
#             return {
#                 "data": result,
#                 "crawl_time_seconds": 0,
#                 "cached": True,
#                 "status": 0
#             }
        
#         # Nếu không có cache, gửi task vào RabbitMQ
#         logger.info(f"Gửi task crawl vào queue cho mst: {tax_no}")
#         request_id = await producer.publish_crawl_task(tax_no)
        
#         return {
#             "message": "Task đã được gửi vào queue",
#             "request_id": request_id,
#             "status": "processing",
#             "poll_url": f"/status/{tax_no}"
#         }
        
#     except Exception as e:
#         logger.error(f"Error searching for tax code {tax_no}: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm: {str(e)}")

# @app.get("/status-queue/{tax_no}")
# async def get_status(tax_no: str):
    # """Kiểm tra trạng thái của request"""
    # try:
    #     redis = await get_redis()
    #     result_key = f"{tax_no}"
    #     result_data = await redis.get(result_key)
        
    #     if result_data:
    #         result = json.loads(result_data)
    #         return {
    #             "tax_no": tax_no,
    #             "status": "completed",
    #             "data": result.get("data"),
    #             "crawl_time_seconds": result.get("crawl_time_seconds"),
    #             "cached": False
    #         }
    #     else:
    #         return {
    #             "tax_no": tax_no,
    #             "status": "processing",
    #             "message": "Task đang được xử lý..."
    #         }
            
    # except Exception as e:
    #     logger.error(f"Error getting status for tax_no {tax_no}: {str(e)}")
    #     raise HTTPException(status_code=500, detail=f"Lỗi khi kiểm tra trạng thái: {str(e)}")

@app.get("/cache")
async def delete_cache(tax_no: str):
    """Xóa cache theo mã số thuế"""
    try:
        redis = await get_redis()
        cache_key = f"{tax_no}"
        deleted = await redis.delete(cache_key)
        if deleted:
            return {"message": f"Đã xóa cache cho mã số thuế: {tax_no}"}
        else:
            return {"message": f"Không tìm thấy cache cho mã số thuế: {tax_no}"}
    except Exception as e:
        logger.error(f"Error deleting cache for tax code {tax_no}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa cache: {str(e)}")
        
@app.get("/close-driver")
async def close_driver():
    """Đóng Chrome driver để giải phóng RAM"""
    try:
        if hasattr(app.state, 'crawler') and app.state.crawler:
            app.state.crawler.close_driver()
            return {"message": "Chrome driver đã được đóng"}
        else:
            return {"message": "Không có driver nào đang chạy"}
    except Exception as e:
        logger.error(f"Error closing driver: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi đóng driver: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower()
    ) 