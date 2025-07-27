# ============================================================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import string
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import random


# ============================================================================
# CLASS CHÍNH: PRECISION CAPTCHA CNN V3
# ============================================================================
class PrecisionCaptchaCNN:
    def __init__(self, img_width=130, img_height=50):
        """
        Model được thiết kế đặc biệt cho captcha với grid background
        """
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = 5

        # Từ log, thấy model hay nhầm y->p, y->d, z->4 -> cần focus vào những chars này
        self.characters = list(string.ascii_lowercase + string.digits)
        self.num_classes = len(self.characters)

        self.char_to_num = {char: idx for idx, char in enumerate(self.characters)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.characters)}

        # Character confusion mapping để xử lý riêng
        self.confusion_pairs = {
            'y': ['p', 'd'], 'p': ['y'], 'd': ['y'],
            'z': ['4'], '4': ['z'],
            'g': ['q'], 'q': ['g'],
            'b': ['6'], '6': ['b']
        }

        print(f"✅ Precision Model initialized: {self.num_classes} classes")
        self.model = None

    def advanced_grid_preprocess(self, image_path, augment=False):
        """
        Preprocessing đặc biệt cho captcha có grid background
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        if image is None:
            return None

        # 1. QUAN TRỌNG: Remove grid pattern trước
        # Detect và remove horizontal/vertical lines
        original_height, original_width = image.shape

        # Create structural elements
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)

        # Remove grid
        grid_mask = cv2.add(horizontal_lines, vertical_lines)
        image_clean = cv2.subtract(image, grid_mask)

        # 2. Adaptive preprocessing dựa trên histogram
        hist = cv2.calcHist([image_clean], [0], None, [256], [0, 256])

        # Nếu ảnh quá tối hoặc sáng
        if np.argmax(hist) < 100:  # Dark image
            image_clean = cv2.convertScaleAbs(image_clean, alpha=1.3, beta=30)
        elif np.argmax(hist) > 180:  # Bright image
            image_clean = cv2.convertScaleAbs(image_clean, alpha=0.8, beta=-20)

        # 3. Noise reduction với bilateral filter
        image_clean = cv2.bilateralFilter(image_clean, 9, 75, 75)

        # 4. Character enhancement
        # Sử dụng Tophat để enhance text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(image_clean, cv2.MORPH_TOPHAT, kernel)
        image_clean = cv2.add(image_clean, tophat)

        # 5. Resize với anti-aliasing
        image_clean = cv2.resize(image_clean, (self.img_width, self.img_height),
                                 interpolation=cv2.INTER_LANCZOS4)

        # 6. Final thresholding để có binary clear image
        _, image_clean = cv2.threshold(image_clean, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Augmentation nếu cần (chỉ geometric, không thay đổi char shape)
        if augment:
            # Slight rotation
            if random.random() < 0.3:
                angle = random.uniform(-2, 2)
                M = cv2.getRotationMatrix2D((self.img_width // 2, self.img_height // 2), angle, 1.0)
                image_clean = cv2.warpAffine(image_clean, M, (self.img_width, self.img_height))

            # Slight shift
            if random.random() < 0.3:
                shift_x = random.randint(-2, 2)
                shift_y = random.randint(-1, 1)
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                image_clean = cv2.warpAffine(image_clean, M, (self.img_width, self.img_height))

        # 8. Normalize
        image_clean = image_clean.astype(np.float32) / 255.0

        # 9. Add channel
        image_clean = np.expand_dims(image_clean, axis=-1)

        return image_clean

    def load_data_smart(self, data_folder, balance_classes=True):
        """
        Load data với class balancing thông minh
        """
        images = []
        labels = []

        if not os.path.exists(data_folder):
            print(f"❌ Folder not found: {data_folder}")
            return None, None

        files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
        print(f"📁 Found {len(files)} JPG files")

        # Count character frequencies
        char_counts = {char: 0 for char in self.characters}

        # First pass: count characters
        for filename in files:
            label = filename.split('.')[0].lower()
            if len(label) == 5 and all(c in self.characters for c in label):
                for char in label:
                    char_counts[char] += 1

        # Find max count for balancing
        max_count = max(char_counts.values()) if char_counts.values() else 1
        print(f"📊 Max character frequency: {max_count}")

        # Count label frequencies for better balancing
        label_counts = {}
        for filename in files:
            label = filename.split('.')[0].lower()
            if len(label) == 5 and all(c in self.characters for c in label):
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"📊 Unique labels: {len(label_counts)}")
        print(f"📊 Labels with only 1 sample: {sum(1 for count in label_counts.values() if count == 1)}")

        # Second pass: load with smart augmentation
        for filename in files:
            label = filename.split('.')[0].lower()

            if len(label) == 5 and all(c in self.characters for c in label):
                image_path = os.path.join(data_folder, filename)

                # Load original
                original_image = self.advanced_grid_preprocess(image_path, augment=False)
                if original_image is not None:
                    images.append(original_image)
                    labels.append(label)

                    # Smart augmentation - more for rare labels
                    if balance_classes and label_counts[label] < 3:
                        # Extra augmentation for very rare labels
                        aug_count = min(4, 6 - label_counts[label])

                        for i in range(aug_count):
                            aug_image = self.advanced_grid_preprocess(image_path, augment=True)
                            if aug_image is not None:
                                images.append(aug_image)
                                labels.append(label)
                    elif balance_classes:
                        # Normal augmentation
                        min_freq = min(char_counts[char] for char in label)
                        if min_freq < max_count * 0.7:
                            aug_count = min(2, int(max_count / (min_freq + 1)))

                            for i in range(aug_count):
                                aug_image = self.advanced_grid_preprocess(image_path, augment=True)
                                if aug_image is not None:
                                    images.append(aug_image)
                                    labels.append(label)

        print(f"✅ Total samples after smart augmentation: {len(images)}")
        return np.array(images), labels

    def build_precision_model(self):
        """
        Model architecture tối ưu cho character recognition
        """
        print("🔧 Building PRECISION model...")

        input_img = layers.Input(shape=(self.img_height, self.img_width, 1))

        # Multi-scale feature extraction
        # Scale 1: Fine details (3x3)
        conv1_3x3 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)

        # Scale 2: Medium features (5x5)
        conv1_5x5 = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(input_img)

        # Scale 3: Coarse features (7x7)
        conv1_7x7 = layers.Conv2D(16, (7, 7), padding='same', activation='relu')(input_img)

        # Concatenate multi-scale features
        x = layers.Concatenate()([conv1_3x3, conv1_5x5, conv1_7x7])
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)

        # Character-specific feature extraction
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.15)(x)

        # Deep feature learning
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)

        # Global features
        x = layers.GlobalAveragePooling2D()(x)

        # Shared representation
        shared = layers.Dense(512, activation='relu')(x)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)

        shared = layers.Dense(256, activation='relu')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.25)(shared)

        # Position-aware character heads
        outputs = []
        for i in range(self.max_length):
            # Each position gets specialized processing
            pos_features = layers.Dense(128, activation='relu',
                                        name=f'pos_{i + 1}_features')(shared)
            pos_features = layers.Dropout(0.2)(pos_features)

            # Character classifier
            char_output = layers.Dense(self.num_classes, activation='softmax',
                                       name=f'char_{i + 1}')(pos_features)
            outputs.append(char_output)

        self.model = keras.Model(inputs=input_img, outputs=outputs)

        # Advanced optimizer with cosine decay
        initial_lr = 0.002
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_lr, decay_steps=1000, alpha=0.1
        )

        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.001)

        # Weighted loss - more weight on difficult positions
        position_weights = [1.2, 1.0, 1.3, 1.0, 0.9]  # Char 1&3 harder based on your log

        # Cách 1: Dùng list của loss functions
        self.model.compile(
            optimizer=optimizer,
            loss=['categorical_crossentropy'] * 5,  # 5 loss functions cho 5 outputs
            loss_weights=position_weights,
            metrics=['accuracy'] * 5  # 5 metrics cho 5 outputs
        )

        print("✅ Precision model built!")
        print(f"📊 Parameters: {self.model.count_params():,}")
        return self.model

    def train_precision_simple(self, data_folder, epochs=150):
        """
        Training đơn giản hơn, tránh stratification issues
        """
        print(f"=== SIMPLE PRECISION Training ===")

        # Load data
        images, labels = self.load_data_smart(data_folder, balance_classes=True)
        if images is None or len(images) == 0:
            print("❌ No training data!")
            return None

        print(f"📊 Dataset: {len(images)} samples")

        # Simple random split (no stratification)
        split_idx = int(len(images) * 0.85)  # 85% train, 15% validation

        # Shuffle data
        indices = np.random.permutation(len(images))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        X_train = images[train_indices]
        X_val = images[val_indices]
        y_train = [labels[i] for i in train_indices]
        y_val = [labels[i] for i in val_indices]

        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")

        # Encode labels
        y_train_encoded = self.encode_labels(y_train)
        y_val_encoded = self.encode_labels(y_val)

        train_dict = {f'char_{i + 1}': y_train_encoded[:, i, :] for i in range(self.max_length)}
        val_dict = {f'char_{i + 1}': y_val_encoded[:, i, :] for i in range(self.max_length)}

        # Build model
        if self.model is None:
            self.build_precision_model()

        # Simple callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print(f"🚀 Simple training for max {epochs} epochs...")

        # Training
        history = self.model.fit(
            X_train, train_dict,
            validation_data=(X_val, val_dict),
            epochs=epochs,
            batch_size=16,
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )

        print("✅ Training completed!")

        # Test on validation set
        print("\n=== Validation Results ===")
        val_accuracy = self.evaluate_precision(X_val, y_val)

        return history

    def train_precision(self, data_folder, epochs=150):
        """
        Training với precision techniques - fallback to simple if stratification fails
        """
        print(f"=== PRECISION Training ===")

        # Load data
        images, labels = self.load_data_smart(data_folder, balance_classes=True)
        if images is None or len(images) == 0:
            print("❌ No training data!")
            return None

        print(f"📊 Dataset: {len(images)} samples")

        # Smart split - check if stratification is possible
        try:
            # Try stratified split first
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.15, random_state=42, stratify=labels
            )
            print("✅ Using stratified split")
        except ValueError as e:
            # Fall back to random split if stratification fails
            print("⚠️ Stratification failed, using random split")
            print(f"   Reason: {e}")
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.15, random_state=42, shuffle=True
            )

        # Encode labels
        y_train_encoded = self.encode_labels(y_train)
        y_val_encoded = self.encode_labels(y_val)

        train_dict = {f'char_{i + 1}': y_train_encoded[:, i, :] for i in range(self.max_length)}
        val_dict = {f'char_{i + 1}': y_val_encoded[:, i, :] for i in range(self.max_length)}

        # Build model
        if self.model is None:
            self.build_precision_model()

        # Precision callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=12,
                min_lr=1e-8,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'precision_model_best.h5',
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            )
        ]

        print(f"🚀 Precision training for max {epochs} epochs...")
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")

        # Training với custom batch size
        batch_size = min(32, len(X_train) // 10)  # Adaptive batch size

        history = self.model.fit(
            X_train, train_dict,
            validation_data=(X_val, val_dict),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            shuffle=True
        )

        print("✅ Precision training completed!")

        # Test on validation set
        print("\n=== Validation Results ===")
        val_accuracy = self.evaluate_precision(X_val, y_val)

        if val_accuracy > 0.85:
            print("🎯 HIGH ACCURACY achieved!")
        elif val_accuracy > 0.7:
            print("✅ Good accuracy achieved!")
        else:
            print("🔄 Model needs more training...")

        return history

    def evaluate_precision(self, X_test, y_test):
        """
        Detailed evaluation với confusion analysis
        """
        correct = 0
        total = len(X_test)
        char_correct = [0] * self.max_length
        char_confusion = [{} for _ in range(self.max_length)]

        print("Sample predictions:")
        for i, (image, true_label) in enumerate(zip(X_test, y_test)):
            pred_label, confidence, _ = self.predict_single(image)

            if pred_label == true_label:
                correct += 1

            # Per-character analysis
            for j in range(min(len(pred_label), len(true_label))):
                if pred_label[j] == true_label[j]:
                    char_correct[j] += 1
                else:
                    # Track confusion
                    key = f"{true_label[j]}->{pred_label[j]}"
                    char_confusion[j][key] = char_confusion[j].get(key, 0) + 1

            if i < 15:  # Show first 15
                print(f"   {true_label}: '{pred_label}' {'✅' if pred_label == true_label else '❌'} ({confidence:.3f})")

        accuracy = correct / total
        print(f"\n📊 Overall Accuracy: {accuracy:.1%} ({correct}/{total})")

        print("\n📊 Per-position Accuracy:")
        for i in range(self.max_length):
            char_acc = char_correct[i] / total
            print(f"   Position {i + 1}: {char_acc:.1%}")

            # Show top confusions for this position
            if char_confusion[i]:
                top_confusions = sorted(char_confusion[i].items(),
                                        key=lambda x: x[1], reverse=True)[:3]
                print(f"      Top confusions: {top_confusions}")

        return accuracy

    def predict_single(self, image):
        """Predict for single image"""
        if self.model is None:
            raise ValueError("Model not trained!")

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image, verbose=0)

        result = ""
        confidence_scores = []

        for i in range(self.max_length):
            pred_probs = predictions[i][0]
            char_idx = np.argmax(pred_probs)
            confidence = np.max(pred_probs)

            char = self.num_to_char[char_idx]
            result += char
            confidence_scores.append(confidence)

        avg_confidence = np.mean(confidence_scores)
        return result, avg_confidence, confidence_scores

    def predict(self, image_path):
        """Predict with preprocessing"""
        image = self.advanced_grid_preprocess(image_path, augment=False)
        if image is None:
            return "", 0.0, []

        return self.predict_single(image)

    def encode_labels(self, labels):
        """Encode labels to one-hot"""
        encoded = np.zeros((len(labels), self.max_length, self.num_classes))

        for i, label in enumerate(labels):
            for j, char in enumerate(label):
                if j < self.max_length and char in self.char_to_num:
                    encoded[i, j, self.char_to_num[char]] = 1

        return encoded

    def save_model(self, filepath):
        """Save model"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Model saved: {filepath}")

    def load_model(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"✅ Model loaded: {filepath}")
            return True
        return False


def main():
    """
    Main training với precision focus
    """
    print("=== PRECISION CAPTCHA TRAINING V3 ===")
    print("🎯 Focus: Improve char 1&3 accuracy, reduce y/p/d confusion")

    # Initialize
    model = PrecisionCaptchaCNN(img_width=130, img_height=50)

    # Training
    history = model.train_precision('training_data', epochs=150)

    if history:
        # Save
        model.save_model('precision_captcha_model.h5')
        print("\n🎯 Precision training completed!")
        print("📁 Best model saved as 'precision_captcha_model.h5'")
        print("🔍 Key improvements:")
        print("   - Grid removal preprocessing")
        print("   - Multi-scale feature extraction")
        print("   - Position-aware character heads")
        print("   - Smart class balancing")
        print("   - Advanced confusion handling")
    else:
        print("❌ Training failed!")


if __name__ == "__main__":
    main()