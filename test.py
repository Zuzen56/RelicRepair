from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# 使用PIL库加载图片
image = Image.open("test2.png")

# 将图片转换为RGB格式（如果不是的话）
image = image.convert("RGB")

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 对图片进行预处理
input_image = preprocess(image).unsqueeze(0)  # 添加一个维度，表示batch size为1


# 定义卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x


# 创建卷积网络实例
conv_net = ConvNet()

# 应用卷积网络
output_feature_map = conv_net(input_image)

# 查看输出的特征图大小
print(output_feature_map.size())