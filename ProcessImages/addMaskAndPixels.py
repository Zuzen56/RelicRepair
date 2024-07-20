import cv2
import numpy as np
import glob
import os
import random
from PIL import Image, ImageFilter

def apply_glitch(image, intensity=1):
    width, height = image.size
    data = image.load()
    for y in range(height):
        for x in range(width):
            if random.random() < 0.1 * intensity:
                data[x, y] = (0, 0, 0)

input_folder_cv2 = '../data/trainTarget'  # 输入文件夹路径（cv2）
output_folder_cv2 = '../data/trainmiddle'  # 输出文件夹路径（cv2）
mask_folder = '../data/mask'  # 存放掩膜图片的文件夹路径

mask_images = glob.glob(os.path.join(mask_folder, '*.jpg')) + glob.glob(os.path.join(mask_folder, '*.png'))

if not mask_images:
    print("没有在'{}'文件夹中找到任何掩膜图片。".format(mask_folder))
else:
    input_images_cv2 = glob.glob(os.path.join(input_folder_cv2, '*.jpg')) + glob.glob(os.path.join(input_folder_cv2, '*.png'))

    for img_path in input_images_cv2:
        img = cv2.imread(img_path)
        chosen_mask_path = random.choice(mask_images)
        damage_img = cv2.imread(chosen_mask_path)

        r1, c1, ch1 = img.shape
        r2, c2, ch2 = damage_img.shape
        roi = img[r1 - r2:r1, c1 - c2:c1]

        gray = cv2.cvtColor(damage_img, cv2.COLOR_BGR2GRAY)
        ret, ma1 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        fg1 = cv2.bitwise_and(roi, roi, mask=ma1)

        ret, ma2 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        fg2 = cv2.bitwise_and(damage_img, damage_img, mask=ma2)

        roi[:] = cv2.add(fg1, fg2)

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder_cv2, filename)
        cv2.imwrite(output_path, img)

    print("OpenCV处理完成！")

input_folder_pil = '../data/trainmiddle'  # 输入文件夹路径（PIL）
output_folder_pil = '../data/trainInput'  # 输出文件夹路径（PIL）
intensity_pil = 1  # 控制破损的程度（PIL）

for filename in os.listdir(input_folder_pil):
    if filename.endswith(".png"):
        img_path = os.path.join(input_folder_pil, filename)
        img = Image.open(img_path)
        apply_glitch(img, intensity_pil)
        output_path = os.path.join(output_folder_pil, filename)
        img.save(output_path)

print("PIL处理完成！")