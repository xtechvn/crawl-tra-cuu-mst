import time
import os
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import logging
from config import config
from utils import predictCaptcha
from utils import submit_tax_form
import requests
from PIL import Image
from io import BytesIO
import time
import re
import json
import psutil  # Thêm để monitor RAM
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
logger = logging.getLogger(__name__)

class TaxCrawler:
    def __init__(self):
        self.base_url = config.CRAWLER_BASE_URL
        self.session = None
        self.driver = None
        self.ua = UserAgent()
        self._driver_initialized = False  # Flag để kiểm tra driver đã khởi tạo chưa
        self._last_used = time.time()  # Track thời gian sử dụng cuối
        self._max_idle_time = 300  # 5 phút không dùng thì đóng  
    

    """
    Tìm kiếm thông tin doanh nghiệp theo mã số thuế
    
    Args:
        tax_code (str): Mã số thuế cần tìm kiếm
        
    Returns:
        Dict[str, Any]: Thông tin doanh nghiệp
    """
    def _init_driver(self):
        """Khởi tạo Chrome driver một lần duy nhất"""
        current_time = time.time()
        
        # Kiểm tra nếu driver đã quá cũ thì đóng và tạo mới
        if (self._driver_initialized and self.driver and 
            current_time - self._last_used > self._max_idle_time):
            logger.info("🔄 Driver quá cũ, đóng và tạo mới...")
            self.close_driver()
        
        if self._driver_initialized and self.driver:
            # Kiểm tra xem driver còn hoạt động không
            try:
                # Test connection bằng cách lấy title
                self.driver.title
                self._last_used = current_time  # Update thời gian sử dụng
                return  # Đã khởi tạo rồi
            except Exception as e:
                logger.warning(f"Driver không còn hoạt động: {e}")
                self.close_driver()  # Đóng driver cũ
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument(f"--window-size={config.CHROME_WINDOW_SIZE}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Thêm options để giảm RAM usage
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")  # Tắt load ảnh để giảm RAM
        chrome_options.add_argument("--disable-javascript")  # Tắt JS nếu không cần
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-features=TranslateUI")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--max_old_space_size=128")  # Giới hạn RAM cho Chrome
        # Thêm options để tăng tốc độ
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-threaded-animation")
        chrome_options.add_argument("--disable-threaded-scrolling")
        chrome_options.add_argument("--disable-checker-imaging")
        chrome_options.add_argument("--disable-new-content-rendering-timeout")
        chrome_options.add_argument("--disable-hang-monitor")
        chrome_options.add_argument("--disable-prompt-on-repost")
        chrome_options.add_argument("--disable-client-side-phishing-detection")
        chrome_options.add_argument("--disable-component-update")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-sync")
        chrome_options.add_argument("--disable-translate")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--no-default-browser-check")
        chrome_options.add_argument("--disable-logging")
        chrome_options.add_argument("--silent")
        chrome_options.add_argument("--log-level=3")

        try:
            # Sử dụng webdriver-manager để tự động tải ChromeDriver phù hợp
            chrome_service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
            # Tối ưu page load strategy
            self.driver.set_page_load_timeout(10)
            self.driver.implicitly_wait(2)  # Giảm implicit wait
            self.driver.get(self.base_url)
            self._driver_initialized = True
            self._last_used = time.time()  # Update thời gian khởi tạo
            logger.info(f"✅ Chrome driver initialized (auto): {self.base_url}")
        except Exception as e:
            logger.error(f"ChromeDriverManager failed: {str(e)}")
            # Fallback to manual chromedriver.exe
            try:
                chromedriver_path = "./chromedriver.exe"
                if os.path.exists(chromedriver_path):
                    chrome_service = Service(chromedriver_path)
                    self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
                    self.driver.get(self.base_url)
                    self._driver_initialized = True
                    self._last_used = time.time()  # Update thời gian khởi tạo
                    logger.info(f"✅ Chrome driver initialized (manual): {self.base_url}")
                else:
                    raise Exception("Không tìm thấy file chromedriver.exe")
            except Exception as e2:
                logger.error(f"Manual ChromeDriver failed: {str(e2)}")
                raise Exception(f"Không thể khởi tạo Chrome WebDriver: {str(e2)}")

    def close_driver(self):
        """Đóng driver khi cần"""
        if self.driver:
            try:
                # Log RAM usage trước khi đóng
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"💾 RAM usage trước khi đóng: {memory_mb:.1f} MB")
                
                self.driver.quit()
                self._driver_initialized = False
                self.driver = None
                
                # Log RAM usage sau khi đóng
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                logger.info(f"✅ Chrome driver closed. RAM sau: {memory_mb_after:.1f} MB (giảm: {memory_mb - memory_mb_after:.1f} MB)")
            except Exception as e:
                logger.error(f"Error closing driver: {str(e)}")

    async def search_by_tax_code(self, tax_code: str) -> (int, dict):
        """
        Chuẩn hóa đầu ra: trả về (status, nnt_json)
        status = 0: thành công, nnt_json là dict dữ liệu
        status != 0: thất bại, nnt_json là None hoặc dict thông tin lỗi
        """
        try:
            # 1. Khởi tạo driver (chỉ 1 lần)
            self._init_driver()
            
            # Giảm thời gian chờ để tối ưu tốc độ
            time.sleep(0.5)  # Giảm từ 1s xuống 0.5s           

            max_attempts = config.CRAWLER_RETRY_COUNT
            for attempt in range(max_attempts):
                # 2. Đọc mã captcha từ ảnh - giảm timeout
                captcha_img = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, '//img[contains(@src, "captcha.png")]'))
                )
                captcha_img_bytes = captcha_img.screenshot_as_png

                pil_image = Image.open(BytesIO(captcha_img_bytes))
                captcha_array = np.array(pil_image.convert('L'))  # Convert to grayscale
                
                logger.info(f"Đã lấy captcha từ memory, shape: {captcha_array.shape}")

                # 2. Dịch captcha bằng model precision_model_best.h5
                start_captcha_time = time.time()
                captcha_code = predictCaptcha(captcha_array, 'precision_model_best.h5')
                end_captcha_time = time.time()
                print(f"captcha dich ra la: {captcha_code} (thời gian dịch: {end_captcha_time - start_captcha_time:.2f}s)")

                # 3. Request url: https://tracuunnt.gdt.gov.vn/tcnnt/mstdn.jsp với các tham số
                cookies = self.driver.get_cookies()
                session = requests.Session()
                for cookie in cookies:
                    session.cookies.set(cookie['name'], cookie['value'])

                # Giả lập truyền mã số thuế và click button
                mst_input = self.driver.find_element(By.NAME, "mst")
                mst_input.clear()
                mst_input.send_keys(tax_code)

                mst_input = self.driver.find_element(By.NAME, "captcha")
                mst_input.clear()
                mst_input.send_keys(captcha_code)

                # Submit form bằng JavaScript để tăng tốc độ
                self.driver.execute_script("search();")

                # Giảm thời gian chờ response
                time.sleep(0.5)  # Giảm thêm
                html = self.driver.page_source

                # 4. Lấy dữ liệu json từ biến nntJson trong html trả về
                match = re.search(r"var\s+nntJson\s*=\s*(\{.*?\});", html, re.DOTALL)
                if match:
                    nnt_json_str = match.group(1)
                    nnt_json = json.loads(nnt_json_str)
                    logger.info("Tìm thấy nnt json")
                    # Kiểm tra nếu status = 0 thì return 1
                    if nnt_json.get("status") == 0:
                        return 1, nnt_json
                    return 0, nnt_json
                else:
                    logger.warning(f"Attempt {attempt+1}: Không tìm thấy dữ liệu nntJson trong response. Thử lại...")
                    if attempt < max_attempts - 1:
                        logger.info(f"Thử lại lần {attempt+1}")
                        self.driver.get(self.base_url)
                    else:
                        # Trả về status lỗi, nnt_json = None
                        return 1, "Không tìm thấy dữ liệu nntJson trong response"

        except Exception as e:
            logger.error(f"Error in search_by_tax_code: {str(e)}")
            return 2, {"error": str(e)}
        finally:
            # KHÔNG đóng driver để reuse cho lần sau
            pass