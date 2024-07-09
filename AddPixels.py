from PIL import Image, ImageFilter
import os
import random

# 检查输出文件夹是否存在，如果不存在则创建
output_folder = 'AddPixels_images'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 读取文件夹中的每张图片
input_folder = 'cut_image'
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 确保只处理图片文件
        input_path = os.path.join(input_folder, filename)

        # 打开图像文件
        img = Image.open(input_path)

        # 定义破损函数
        def apply_glitch(image, intensity=1):
            width, height = image.size
            data = image.load()
            for _ in range(int(width * height * 0.1 * intensity)):  # 根据强度计算应该添加的黑色像素点数量
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                data[x, y] = (0, 0, 0)  # 随机将像素点设为黑色

        # 添加破损效果
        apply_glitch(img, intensity=1)  # 这里的intensity可以控制破损的程度


        # 保存处理后的图片
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)

        print(f"Processed {filename}")

print("All images processed and saved.")