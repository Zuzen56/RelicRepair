import torch
from NetModel import Net
from PIL import Image
import torchvision.transforms as transforms

# 加载模型结构
net = Net()

# 加载模型参数
pthfile = 'model_best.pth'
net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))  # 如果在 CPU 上运行的话需要指定 map_location=torch.device('cpu')
net.eval()

# 读取待处理的图片
input_image = Image.open('input.png').convert('RGB')

# 定义预处理的操作
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整输入大小
    transforms.ToTensor(),
])

# 对图片进行预处理
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # 添加 batch 维度

# 使用模型进行预测
with torch.no_grad():
    output = net(input_batch)

# 定义后处理的操作
postprocess = transforms.Compose([
    # 需要的后处理操作，比如将张量转换为图片
    transforms.ToPILImage()
])

# 对预测结果进行后处理
output_image = output[0]  # 假设这里的output是图片的张量表示，根据实际情况进行调整
output_image = postprocess(output_image)

# 保存预测结果
output_image.save('output.png')


