from PIL import Image, ImageFilter
import random
import os

# 遍历input文件夹中的所有图片
input_folder = 'trainInput'
output_folder = 'trainInput'
intensity = 1  # 控制破损的程度

# 定义破损函数
def apply_glitch(image, intensity=1):
    width, height = image.size
    data = image.load()
    for y in range(height):
        for x in range(width):
            if random.random() < 0.1 * intensity:  # 以10%的概率应用破损效果
                data[x, y] = (0, 0, 0)  # 将像素点设为黑色

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # 读取原始图片
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # 应用破损效果
        apply_glitch(img, intensity)

        # 保存处理后的图片到output文件夹
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)

print("处理完成！")