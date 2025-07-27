import requests
import time


def submit_tax_form(mst="", fullname="", address="", cmt="", captcha_text=""):
      # Setup session
    session = requests.Session()
    base_url = "http://tracuunnt.gdt.gov.vn"

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Referer': f'{base_url}/tcnnt/mstcn.jsp'
    })

    try:
        # Bước 1: Get form page để có session
        form_response = session.get(f"{base_url}/tcnnt/mstcn.jsp")
        if form_response.status_code != 200:
            return {
                'success': False,
                'content': '',
                'message': f'Không thể truy cập form page: {form_response.status_code}',
                'status_code': form_response.status_code
            }

        # Bước 2: Chuẩn bị form data
        form_data = {
            'cm': 'cm',
            'mst': mst,
            'fullname': fullname,
            'address': address,
            'cmt': cmt,
            'captcha': captcha_text
        }
        print(form_data)
        # Bước 3: Submit form
        submit_response = session.post(
            f"{base_url}/tcnnt/mstcn.jsp",
            data=form_data,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        )
        content = submit_response.text
        return {
            'success': True,
            'content': content
        }

    except Exception as e:
        return {
            'success': False,
            'content': '',
            'message': f'Lỗi: {str(e)}',
            'status_code': 0
        }