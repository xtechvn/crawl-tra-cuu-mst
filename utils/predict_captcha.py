import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import string
import logging

# Disable TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predictCaptcha(
        image_input,  # Có thể là file path hoặc numpy array
        model_path,
        img_width=130,
        img_height=50,
        max_length=5  # FIXED: 5 characters, not 7
):
    """
    Dịch captcha từ ảnh sử dụng model đã train (.h5)

    Args:
        image_input: Đường dẫn ảnh captcha (str) hoặc numpy array
        model_path (str): Đường dẫn model .h5
        img_width (int): Chiều rộng ảnh đầu vào model
        img_height (int): Chiều cao ảnh đầu vào model
        max_length (int): Số ký tự captcha

    Returns:
        result (str): Chuỗi captcha dự đoán
    """
    try:
        # Danh sách ký tự: a-z + 0-9
        characters = list(string.ascii_lowercase + string.digits)
        num_to_char = {idx: char for idx, char in enumerate(characters)}

        # Validate model file
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return ""

        # Handle image input (file path or numpy array)
        if isinstance(image_input, str):
            # File path
            if not os.path.exists(image_input):
                print(f"❌ Image file not found: {image_input}")
                return ""
            image_path = image_input
        elif isinstance(image_input, np.ndarray):
            # Numpy array - use directly
            image = image_input
            print("🖼️ Using numpy array input")
        else:
            print(f"❌ Invalid image input type: {type(image_input)}")
            return ""

        # Load model with error handling
        try:
            print(f"🧠 Loading model: {model_path}")
            model = keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Cannot load model: {e}")
            return ""

        # Tiền xử lý ảnh
        if isinstance(image_input, str):
            # Load from file
            print(f"🖼️ Processing image file: {image_path}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"❌ Cannot read image: {image_path}")
                return ""
        # Nếu là numpy array thì đã có image rồi

        # Basic preprocessing
        image = cv2.resize(image, (img_width, img_height))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)
        image = np.expand_dims(image, axis=0)  # (1, H, W, 1)

        print(f"📐 Input shape: {image.shape}")

        # Dự đoán với error handling
        try:
            print("🎯 Making prediction...")
            predictions = model.predict(image, verbose=0)
            print("✅ Prediction successful")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return ""

        # Xử lý kết quả predictions
        result = ""

        # FIXED: Handle different prediction formats
        if isinstance(predictions, list):
            # Multi-output model (5 separate outputs)
            print(f"📊 Multi-output model: {len(predictions)} outputs")

            for i in range(min(max_length, len(predictions))):
                try:
                    pred_probs = predictions[i][0]  # Get first batch
                    char_idx = np.argmax(pred_probs)
                    confidence = np.max(pred_probs)

                    if char_idx < len(characters):
                        char = num_to_char[char_idx]
                        result += char
                        print(f"   Pos {i + 1}: '{char}' ({confidence:.3f})")
                    else:
                        print(f"⚠️ Invalid char index {char_idx} at position {i + 1}")
                        result += "?"

                except (IndexError, KeyError) as e:
                    print(f"❌ Error processing position {i + 1}: {e}")
                    result += "?"

        elif isinstance(predictions, np.ndarray):
            # Single output model
            print(f"📊 Single output shape: {predictions.shape}")

            if len(predictions.shape) == 3:  # (batch, sequence, classes)
                seq_len = min(max_length, predictions.shape[1])
                for i in range(seq_len):
                    try:
                        pred_probs = predictions[0, i, :]
                        char_idx = np.argmax(pred_probs)
                        confidence = np.max(pred_probs)

                        if char_idx < len(characters):
                            char = num_to_char[char_idx]
                            result += char
                            print(f"   Pos {i + 1}: '{char}' ({confidence:.3f})")
                        else:
                            result += "?"

                    except (IndexError, KeyError) as e:
                        print(f"❌ Error at position {i + 1}: {e}")
                        result += "?"
            else:
                print(f"❌ Unexpected prediction shape: {predictions.shape}")
                return ""
        else:
            print(f"❌ Unknown prediction type: {type(predictions)}")
            return ""

        # Ensure correct length
        if len(result) < max_length:
            result = result.ljust(max_length, '?')
        elif len(result) > max_length:
            result = result[:max_length]

        print(f"✅ Final result: '{result}' (length: {len(result)})")
        return result.lower()

    except Exception as e:
        print(f"❌ Unexpected error in predictCaptcha: {e}")
        import traceback
        traceback.print_exc()
        return ""


# Test function
def test_captcha_prediction():
    """Test the prediction function"""
    image_path = "../captcha.jpg"
    model_path = "../precision_model_best.h5"

    if os.path.exists(image_path) and os.path.exists(model_path):
        print("=== TESTING CAPTCHA PREDICTION ===")
        result = predictCaptcha(image_path, model_path)
        print(f"🎯 Test result: '{result}'")
        return result
    else:
        print("❌ Test files not found:")
        print(f"   Image: {image_path} ({'✅' if os.path.exists(image_path) else '❌'})")
        print(f"   Model: {model_path} ({'✅' if os.path.exists(model_path) else '❌'})")
        return ""


if __name__ == "__main__":
    test_captcha_prediction()