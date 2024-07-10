from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# 使用PIL库加载图片
image = Image.open("input_image.png")

# 将图片转换为RGB格式（如果不是的话）
image = image.convert("RGB")

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 对图片进行预处理
input_image = preprocess(image).unsqueeze(0)  # 添加一个维度，表示batch size为1

# 在此之后再进行卷积操作
# 定义卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

# 应用卷积层
output_feature_map = conv_layer(input_image)

# 查看输出的特征图大小
print(output_feature_map.size())