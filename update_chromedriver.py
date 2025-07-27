#!/usr/bin/env python3
"""
Script tự động cập nhật ChromeDriver khi Chrome update
"""
import os
import sys
import requests
import zipfile
import subprocess
from pathlib import Path

def get_chrome_version():
    """Lấy version của Chrome đang cài đặt"""
    try:
        # Windows
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon")
        version, _ = winreg.QueryValueEx(key, "version")
        return version
    except:
        try:
            # Fallback: sử dụng command line
            result = subprocess.run(['google-chrome', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split()[-1]
        except:
            pass
    return None

def get_chromedriver_version(chrome_version):
    """Lấy ChromeDriver version tương ứng với Chrome version"""
    # Lấy major version (ví dụ: 138.0.7204.169 -> 138)
    major_version = chrome_version.split('.')[0]
    
    # Thử từ version cao nhất xuống thấp nhất
    for version in range(int(major_version), int(major_version) - 10, -1):
        url = f"https://chromedriver.storage.googleapis.com/LATEST_RELEASE_{version}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                chromedriver_version = response.text.strip()
                print(f"🔍 Tìm thấy ChromeDriver version {chromedriver_version} cho Chrome {major_version}")
                return chromedriver_version
        except:
            continue
    
    print(f"❌ Không tìm thấy ChromeDriver cho Chrome version {major_version}")
    return None

def download_chromedriver(version):
    """Tải ChromeDriver về"""
    url = f"https://chromedriver.storage.googleapis.com/{version}/chromedriver_win64.zip"
    
    try:
        print(f"📥 Đang tải ChromeDriver version {version}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Lưu file zip
            zip_path = "chromedriver.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Giải nén
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Xóa file zip
            os.remove(zip_path)
            
            print(f"✅ ChromeDriver {version} đã được tải và cài đặt thành công!")
            return True
        else:
            print(f"❌ Không thể tải ChromeDriver version {version}")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi tải ChromeDriver: {e}")
        return False

def main():
    """Main function"""
    print("🔄 Kiểm tra và cập nhật ChromeDriver...")
    
    # Lấy Chrome version
    chrome_version = get_chrome_version()
    if not chrome_version:
        print("❌ Không thể xác định version của Chrome")
        return
    
    print(f"🔍 Chrome version: {chrome_version}")
    
    # Lấy ChromeDriver version tương ứng
    chromedriver_version = get_chromedriver_version(chrome_version)
    if not chromedriver_version:
        print("❌ Không thể xác định ChromeDriver version tương ứng")
        print("💡 Giải pháp:")
        print("   1. Sử dụng webdriver-manager (tự động)")
        print("   2. Tải ChromeDriver thủ công từ: https://chromedriver.chromium.org/")
        print("   3. Hoặc downgrade Chrome xuống version cũ hơn")
        return
    
    print(f"🔍 ChromeDriver version cần thiết: {chromedriver_version}")
    
    # Kiểm tra ChromeDriver hiện tại
    chromedriver_path = Path("./chromedriver.exe")
    if chromedriver_path.exists():
        print("📁 Tìm thấy ChromeDriver hiện tại")
        
        # Kiểm tra version hiện tại
        try:
            result = subprocess.run(['./chromedriver.exe', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                current_version = result.stdout.strip().split()[1]
                print(f"📋 ChromeDriver version hiện tại: {current_version}")
                
                if current_version == chromedriver_version:
                    print("✅ ChromeDriver đã là version mới nhất!")
                    return
                else:
                    print("🔄 Cần cập nhật ChromeDriver...")
        except:
            print("⚠️ Không thể kiểm tra version ChromeDriver hiện tại")
    
    # Tải ChromeDriver mới
    if download_chromedriver(chromedriver_version):
        print("🎉 Cập nhật ChromeDriver thành công!")
    else:
        print("❌ Cập nhật ChromeDriver thất bại!")

if __name__ == "__main__":
    main() 