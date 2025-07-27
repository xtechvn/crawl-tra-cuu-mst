import requests

def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """
    Gửi tin nhắn đến Telegram thông qua Bot API.

    Args:
        bot_token (str): Token của Telegram Bot.
        chat_id (str): ID của chat (có thể là user hoặc group).
        message (str): Nội dung tin nhắn.

    Returns:
        bool: True nếu gửi thành công, False nếu có lỗi.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result.get("ok", False)
    except Exception as e:
        print(f"Lỗi khi gửi tin nhắn Telegram: {e}")
        return False









