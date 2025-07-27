import requests

def send_slack_message(webhook_url: str, message: str) -> bool:
    """
    Gửi tin nhắn tới Slack thông qua Incoming Webhook.

    Args:
        webhook_url (str): URL webhook của Slack.
        message (str): Nội dung tin nhắn.

    Returns:
        bool: True nếu gửi thành công, False nếu có lỗi.
    """
    payload = {
        "text": message
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Lỗi khi gửi tin nhắn Slack: {e}")
        return False
