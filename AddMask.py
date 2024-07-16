import cv2
import numpy as np
import glob
import os
import random

input_folder = 'trainInput'  # 输入文件夹路径
output_folder = 'trainInput'  # 输出文件夹路径
mask_folder = 'mask'  # 存放掩膜图片的文件夹路径

# 获取mask文件夹中的所有掩膜图片路径
mask_images = glob.glob(os.path.join(mask_folder, '*.jpg'))

if not mask_images:
    print("没有在'{}'文件夹中找到任何掩膜图片。".format(mask_folder))
else:
    # 获取输入文件夹中的所有图片路径
    input_images = glob.glob(os.path.join(input_folder, '*.jpg')) + glob.glob(os.path.join(input_folder, '*.png'))

    # 循环处理每张图片
    for img_path in input_images:
        img = cv2.imread(img_path)

        # 随机选择一个掩膜图片
        chosen_mask_path = random.choice(mask_images)
        damage_img = cv2.imread(chosen_mask_path)

        # 应用效果
        r1, c1, ch1 = img.shape
        r2, c2, ch2 = damage_img.shape
        roi = img[r1 - r2:r1, c1 - c2:c1]

        gray = cv2.cvtColor(damage_img, cv2.COLOR_BGR2GRAY)
        ret, ma1 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        fg1 = cv2.bitwise_and(roi, roi, mask=ma1)

        ret, ma2 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        fg2 = cv2.bitwise_and(damage_img, damage_img, mask=ma2)

        roi[:] = cv2.add(fg1, fg2)

        # 保存输出图片
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

    print("处理完成！")