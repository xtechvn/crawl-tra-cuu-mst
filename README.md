# Tax Data Crawler API

Ứng dụng FastAPI để crawl dữ liệu từ trang web tracuunnt.gdt.gov.vn

## Tính năng

- Tìm kiếm thông tin doanh nghiệp theo mã số thuế
- API RESTful với documentation tự động
- Logging chi tiết
- Xử lý lỗi và retry mechanism

## Cài đặt

### 1. Cài đặt dependencies

```bash
# Kích hoạt môi trường ảo (nếu có)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cài đặt Chrome WebDriver

Ứng dụng sẽ tự động tải Chrome WebDriver thông qua `webdriver-manager`. Đảm bảo bạn đã cài đặt Google Chrome trên máy.

## Chạy ứng dụng

### Chạy development server

```bash
python run.py
```

Hoặc sử dụng uvicorn trực tiếp:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Truy cập API

- API Documentation: http://localhost:8000/docs
- Alternative Documentation: http://localhost:8000/redoc
- Root endpoint: http://localhost:8000/

## API Endpoints

### Tìm kiếm thông tin doanh nghiệp

**POST** `/search`

```json
{
  "tax_code": "0123456789",
  "company_name": "Tên công ty (tùy chọn)"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Tìm kiếm thành công",
  "data": {
    "company_info": {
      "name": "Tên doanh nghiệp"
    },
    "tax_info": {
      "tax_code": "0123456789"
    },
    "address_info": {
      "full_address": "Địa chỉ đầy đủ"
    },
    "crawled_at": 1703123456.789,
    "status": "success"
  },
  "timestamp": "2023-12-21T10:30:56.789Z"
}
```

## Cấu trúc dự án

```
├── main.py                 # FastAPI application chính
├── requirements.txt        # Dependencies
├── README.md              # Hướng dẫn sử dụng
├── crawler/
│   ├── __init__.py
│   └── tax_crawler.py     # Class crawler chính
├── utils/
│   ├── __init__.py
│   └── logger.py          # Setup logging
└── logs/                  # Thư mục chứa log files
```

## Sử dụng với Python

```python
import requests
import json

# Tìm kiếm thông tin doanh nghiệp
def search_company(tax_code):
    url = "http://localhost:8000/search"
    data = {"tax_code": tax_code}
    
    response = requests.post(url, json=data)
    return response.json()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tìm kiếm thông tin doanh nghiệp
    result = search_company("0123456789")
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

## Lưu ý

1. **Rate Limiting**: Trang web có thể có giới hạn số lượng request, hãy sử dụng hợp lý
2. **Captcha**: Trang web có thể có captcha, crawler sẽ cố gắng xử lý tự động
3. **Legal Compliance**: Đảm bảo tuân thủ các quy định pháp luật khi crawl dữ liệu
4. **Error Handling**: Ứng dụng có cơ chế xử lý lỗi và retry, nhưng vẫn có thể gặp lỗi do thay đổi cấu trúc website

## Troubleshooting

### Lỗi Chrome WebDriver
- Đảm bảo đã cài đặt Google Chrome
- Kiểm tra phiên bản Chrome và WebDriver có tương thích

### Lỗi kết nối
- Kiểm tra kết nối internet
- Kiểm tra trang web có hoạt động không

### Lỗi parsing dữ liệu
- Kiểm tra log files trong thư mục `logs/`
- Có thể cần cập nhật selectors nếu website thay đổi cấu trúc

## License

MIT License 