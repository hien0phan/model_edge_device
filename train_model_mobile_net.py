import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
callbacks = tf.keras.callbacks
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import glob

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
PATH_PERSON = r"D:\dataset\train\person"       
PATH_NO_PERSON = r"D:\dataset\train\no_person" 

# Config cho K-Fold
K_FOLDS = 5             # Số lượng fold (chia dữ liệu làm 5 phần)
BATCH_SIZE = 32
EPOCHS = 50             # Số epoch tối đa mỗi fold
L2_RATE = 1e-4

# Thông số Model/Input
INPUT_LEN = 768
IMG_HEIGHT = 24
IMG_WIDTH = 32

# ==========================================
# 2. CHUẨN HÓA NÂNG CAO (GIỮ NGUYÊN CHO ESP32)
# ==========================================
def robust_normalize_v2(data):
    """
    Hàm chuẩn hóa tối ưu cho dữ liệu nhiệt thay đổi theo mùa.
    Output: Range [-1, 1] (Tốt nhất cho MobileNet và Quantization INT8)
    """
    # [QUAN TRỌNG 1] Kẹp giá trị (Clipping)
    data = np.clip(data, 10.0, 45.0)
    
    d_min = data.min()
    d_max = data.max()
    diff = d_max - d_min
    
    # [QUAN TRỌNG 2] Ngưỡng lọc nhiễu
    scale = max(diff, 2.5) 
    
    # [QUAN TRỌNG 3] Chuyển về [-1, 1]
    norm_0_1 = (data - d_min) / scale
    
    # Công thức: (x - 0.5) * 2 -> Range [-1, 1]
    return (norm_0_1 - 0.5) * 2.0

# ==========================================
# 3. LOAD TOÀN BỘ DATA (ĐỂ CHIA K-FOLD)
# ==========================================
def load_all_data_from_folder(folder, label):
    """
    Đọc tất cả dữ liệu trong folder, không chia split ở đây.
    """
    print(f"-> Đang quét toàn bộ file trong: {folder}")
    files = glob.glob(os.path.join(folder, "*"))
    files.sort()
    
    arr = []
    valid_count = 0
    
    for f in files:
        try:
            if os.path.isdir(f): continue
            
            # Đọc dữ liệu (Hỗ trợ cả phẩy và khoảng trắng)
            try: raw = np.loadtxt(f, delimiter=',', dtype=np.float32)
            except: raw = np.loadtxt(f, dtype=np.float32)
            
            if raw.size == INPUT_LEN:
                # Gọi hàm chuẩn hóa v2
                norm = robust_normalize_v2(raw.reshape(INPUT_LEN))
                arr.append(norm)
                valid_count += 1
        except: pass
        
    print(f"   -> Đã load được {valid_count} mẫu hợp lệ.")
    X = np.array(arr, dtype=np.float32)
    y = np.full(len(X), label, dtype=np.float32)
    return X, y

# ==========================================
# 4. MODEL ARCHITECTURE (MobileNet-Tiny Custom)
# ==========================================
# Giữ nguyên các hàm custom object để load/save model không bị lỗi
def hard_swish(x):
    return x * layers.ReLU(max_value=6.0)(x + 3.0) / 6.0

def se_block(inputs, filters, ratio=4):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(x)
    return layers.Multiply()([inputs, x])

def bneck(x, expansion, out_filters, kernel_size, stride, use_se, activation_fn):
    input_tensor = x
    if expansion > 0:
        x = layers.Conv2D(expansion, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_RATE))(x)
        x = layers.BatchNormalization()(x)
        if activation_fn == 'hard_swish': x = hard_swish(x)
        else: x = activation_fn(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False, depthwise_regularizer=regularizers.l2(L2_RATE))(x)
    x = layers.BatchNormalization()(x)
    if activation_fn == 'hard_swish': x = hard_swish(x)
    else: x = activation_fn(x)
    if use_se: x = se_block(x, expansion)
    x = layers.Conv2D(out_filters, 1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_RATE))(x)
    x = layers.BatchNormalization()(x)
    if stride == 1 and input_tensor.shape[-1] == out_filters:
        x = layers.Add()([input_tensor, x])
    return x

def build_model():
    inputs = layers.Input(shape=(INPUT_LEN,))
    # Reshape vector thành ảnh 24x32
    x = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(inputs)
    
    # Augmentation
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    # Initial Conv
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(L2_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = hard_swish(x)

    # Bottleneck Blocks
    x = bneck(x, 16, 16, 3, 1, True, layers.ReLU())
    x = bneck(x, 48, 24, 3, 2, False, layers.ReLU()) 
    x = bneck(x, 72, 24, 3, 1, True, layers.ReLU())
    x = bneck(x, 72, 40, 5, 2, True, 'hard_swish')     
    x = bneck(x, 120, 40, 5, 1, True, 'hard_swish')
    x = bneck(x, 120, 48, 5, 1, True, 'hard_swish')

    # Last Stage
    x = layers.Conv2D(288, 1, use_bias=False, kernel_regularizer=regularizers.l2(L2_RATE))(x)
    x = layers.BatchNormalization()(x)
    x = hard_swish(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(L2_RATE))(x)
    
    x = layers.Dropout(0.3)(x) 
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# ==========================================
# 5. MAIN PROGRAM (K-FOLD)
# ==========================================
if __name__ == "__main__":
    print(f"--- TRAIN MODEL VỚI {K_FOLDS}-FOLD CROSS VALIDATION ---")
    
    # 1. Load tất cả dữ liệu
    X_p, y_p = load_all_data_from_folder(PATH_PERSON, 1.0)
    X_np, y_np = load_all_data_from_folder(PATH_NO_PERSON, 0.0)
    
    # Gộp lại
    X_all = np.concatenate((X_p, X_np), axis=0)
    y_all = np.concatenate((y_p, y_np), axis=0)
    
    # Shuffle toàn bộ dataset một lần
    X_all, y_all = shuffle(X_all, y_all, random_state=42)
    
    print(f"\nTổng dữ liệu: {len(X_all)} mẫu.")
    print(f"Range input: {X_all.min():.2f} -> {X_all.max():.2f} (Đã chuẩn hóa)")

    # 2. Thiết lập K-Fold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    
    # Biến lưu thông tin fold tốt nhất để export
    best_fold_acc = 0.0
    best_fold_idx = -1
    best_model_path = "temp_best_model.keras" # Đường dẫn tạm lưu model tốt nhất

    for train_idx, val_idx in kfold.split(X_all, y_all):
        print(f'\n------------------------------------------------------------------------')
        print(f'Training for FOLD {fold_no} ...')
        
        # Chia dữ liệu theo index của fold hiện tại
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        
        # Xóa session cũ để giải phóng RAM
        tf.keras.backend.clear_session()
        
        # Build model mới hoàn toàn cho fold này
        model = build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks cho fold này
        # Lưu model tốt nhất của fold hiện tại vào file tạm
        fold_checkpoint_path = f"model_fold_{fold_no}.keras"
        checkpoint = callbacks.ModelCheckpoint(fold_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, reduce_lr, early_stop],
            verbose=1
        )
        
        # Đánh giá Fold này (Sử dụng weights tốt nhất đã restore từ EarlyStopping/Checkpoint)
        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f'Score for Fold {fold_no}: Loss={scores[0]:.4f} - Accuracy={scores[1]*100:.2f}%')
        
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Kiểm tra xem Fold này có phải tốt nhất từ trước đến giờ không
        if scores[1] > best_fold_acc:
            print(f"--> Fold {fold_no} hiện đang là tốt nhất! (Acc: {scores[1]*100:.2f}%)")
            best_fold_acc = scores[1]
            best_fold_idx = fold_no
            # Sao chép model của fold này ra file riêng để dành export
            model.save(best_model_path)
            
        fold_no += 1

    # 3. Tổng kết K-Fold
    print('\n================================================================--------')
    print('KẾT QUẢ K-FOLD CROSS VALIDATION')
    print(f'Average Accuracy: {np.mean(acc_per_fold):.2f}% (+/- {np.std(acc_per_fold):.2f}%)')
    print(f'Average Loss: {np.mean(loss_per_fold):.4f}')
    print(f'FOLD TỐT NHẤT: Fold {best_fold_idx} với Accuracy {best_fold_acc*100:.2f}%')
    print('================================================================--------')

    # ==========================================
    # 6. EXPORT TFLITE (TỪ FOLD TỐT NHẤT)
    # ==========================================
    print(f"\n--- ĐANG EXPORT TFLITE TỪ FOLD {best_fold_idx} ---")
    
    # Load lại model tốt nhất (cần custom_objects vì dùng hard_swish)
    best_model = models.load_model(best_model_path, custom_objects={'hard_swish': hard_swish})
    
    # Representative dataset generator
    # Lấy mẫu từ toàn bộ dataset để tính toán lượng tử hóa
    def rep_data():
        # Lấy 200 mẫu ngẫu nhiên từ X_all để làm mẫu chuẩn hóa
        for i in range(min(200, len(X_all))):
            yield [X_all[i].reshape(1, INPUT_LEN)]

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    
    # Cấu hình Full Integer Quantization cho ESP32
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    tflite_filename = f'best_model_kfold_acc_{best_fold_acc*100:.1f}.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
        
    print(f"DONE! Đã lưu file: {tflite_filename}")
    print(f"File này đã được tối ưu INT8 để nhúng vào ESP32.")