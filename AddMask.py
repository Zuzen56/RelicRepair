from PIL import Image, ImageDraw
import random
import os

# 输入和输出文件夹路径
input_folder = "D:/PycharmProjects/RelicRepair/testTarget"
output_folder = "D:/PycharmProjects/RelicRepair/testInput"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图像
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        img = Image.open(input_path)

        # 创建一个ImageDraw对象
        draw = ImageDraw.Draw(img)

        # 生成随机的多边形坐标
        num_points = 8
        width, height = img.size
        points = []
        for _ in range(num_points):
            x = random.randint(0, width)
            y = random.randint(0, height)
            points.append((x, y))

        # 绘制不规则多边形
        draw.polygon(points, fill=(0, 0, 0))

        # 保存图像到输出文件夹
        img.save(output_path)

print("操作完成")