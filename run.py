#!/usr/bin/env python3
"""
Script để chạy ứng dụng Tax Crawler API
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config

def main():
    """Main function to run the application"""
    print("🚀 Starting Tax Crawler API...")
    print(f"📍 Host: {config.API_HOST}")
    print(f"🔌 Port: {config.API_PORT}")
    print(f"🔄 Reload: {config.API_RELOAD}")
    print(f"📝 Log Level: {config.LOG_LEVEL}")
    print(f"🌐 Environment: {'development' if config.is_development() else 'production'}")
    print()
    print("📖 API Documentation will be available at:")
    print(f"   http://{config.API_HOST}:{config.API_PORT}/docs")
    print(f"   http://{config.API_HOST}:{config.API_PORT}/redoc")
    print()
    print("🏥 Health Check:")
    print(f"   http://{config.API_HOST}:{config.API_PORT}/health")
    print()
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=config.API_RELOAD,
            log_level=config.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 