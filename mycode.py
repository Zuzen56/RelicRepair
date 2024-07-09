import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
import os

# 设置训练和测试数据集的路径
train_input_dir = 'D:/PycharmProjects/RelicRepair/trainInput'
train_target_dir = 'D:/PycharmProjects/RelicRepair/trainTarget'
test_input_dir = 'D:/PycharmProjects/RelicRepair/testInput'
test_target_dir = 'D:/PycharmProjects/RelicRepair/testTarget'
output_dir = 'D:/PycharmProjects/RelicRepair/Output'

# 加载训练和测试数据集
train_input_files = os.listdir(train_input_dir)
train_target_files = os.listdir(train_target_dir)
test_input_files = os.listdir(test_input_dir)
test_target_files = os.listdir(test_target_dir)

# 构建数据输入流水线
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)  # 假设为RGB图像
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))  # 调整图像形状为期望的大小
    return image

def preprocess_image(input_image, target_image):
    # 对图像进行预处理，如缩放、归一化等
    return input_image, target_image

def create_dataset(input_files, target_files):
    input_paths = [os.path.join(train_input_dir, fname) for fname in input_files]
    target_paths = [os.path.join(train_target_dir, fname) for fname in target_files]
    input_dataset = tf.data.Dataset.from_tensor_slices(input_paths).map(load_image)
    target_dataset = tf.data.Dataset.from_tensor_slices(target_paths).map(load_image)
    dataset = tf.data.Dataset.zip((input_dataset, target_dataset))
    dataset = dataset.map(preprocess_image)
    return dataset

train_dataset = create_dataset(train_input_files, train_target_files)
test_dataset = create_dataset(test_input_files, test_target_files)

# 构建修复模型
def create_model():
    model = tf.keras.Sequential([
        # 编码器部分
        Conv2D(64, 3, activation='relu', padding='same', input_shape=(256, 256, 3)),
        BatchNormalization(),
        Conv2D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same', strides=2),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, 3, activation='relu', padding='same', strides=2),
        BatchNormalization(),
        Conv2D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2),
        BatchNormalization(),
        Conv2D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(3, 3, activation='sigmoid', padding='same')
    ])

    return model

model = create_model()

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(train_dataset, epochs=10)

# 测试模型
test_results = model.evaluate(test_dataset)

# 保存修复模型
model.save('image_restoration_model.h5')

# 使用模型修复受损图片并输出结果
test_input_paths = [os.path.join(test_input_dir, fname) for fname in test_input_files]
restored_images = []
for input_path in test_input_paths:
    input_image = load_image(input_path)
    input_image = preprocess_image(input_image, None)[0]
    restored_image = model.predict(tf.expand_dims(input_image, axis=0))
    restored_images.append(restored_image)

# 将修复结果保存到输出目录
os.makedirs(output_dir, exist_ok=True)
for i, restored_image in enumerate(restored_images):
    output_path = os.path.join(output_dir, f'restored_image_{i}.png')
    tf.keras.preprocessing.image.save_img(output_path, restored_image)