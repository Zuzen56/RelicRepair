import os
import random

import torch
import torchvision.transforms.functional as ttf
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def psnr(output_value,target_value):
    # Assuming output_value and target_value are torch tensors
    output_value_np = output_value.cpu().detach().numpy()
    target_value_np = target_value.cpu().detach().numpy()

    # Calculate the mean squared error (MSE)
    mse = np.mean((output_value_np - target_value_np) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_value = 1.0  # If the pixel values are between 0 and 1

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

    psnr = torch.tensor(psnr)

    return psnr


class MyTrainDataSet(Dataset):  # 训练数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片

        # 定义预处理转换
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 对图片进行预处理
        inputImage = preprocess(inputImage)  # 添加一个维度，表示batch size为1

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        # 对图片进行预处理
        targetImage = preprocess(targetImage)  # 添加一个维度，表示batch size为1


        # #这段应该是用不上了
        # # inputImage = ttf.to_tensor(inputImage)  # 将图片转为张量
        # # targetImage = ttf.to_tensor(targetImage)
        #
        # hh, ww = targetImage.shape[1], targetImage.shape[2]  # 图片的高和宽
        # print(hh,ww)
        # rr = random.randint(0, hh-ps)  # 随机数： patch 左下角的坐标 (rr, cc)
        # cc = random.randint(0, ww-ps)
        # # aug = random.randint(0, 8)  # 随机数，对应对图片进行的操作
        #
        # input_ = inputImage[:, rr:rr+ps, cc:cc+ps]  # 裁剪 patch ，输入和目标 patch 要对应相同
        # target = targetImage[:, rr:rr+ps, cc:cc+ps]
        #
        # return input_, target

        return inputImage,targetImage

class MyValueDataSet(Dataset):  # 评估数据集
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=128):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表

        self.ps = patch_size

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片,灰度图

        # 定义预处理转换
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 对图片进行预处理
        inputImage = preprocess(inputImage) # 添加一个维度，表示batch size为1

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        # 对图片进行预处理
        targetImage = preprocess(targetImage) # 添加一个维度，表示batch size为1

        # inputImage = ttf.center_crop(inputImage, (ps, ps))
        # targetImage = ttf.center_crop(targetImage, (ps, ps))

        # input_ = ttf.to_tensor(inputImage)  # 将图片转为张量
        # target = ttf.to_tensor(targetImage)

        return inputImage,targetImage
        #return input_, target

class MyTestDataSet(Dataset):  # 测试数据集
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)  # 输入图片路径下的所有文件名列表

    def __len__(self):
        return len(self.inputImages)  # 路径里的图片数量

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片

        # 定义预处理转换
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # 对图片进行预处理
        inputImage = preprocess(inputImage)  # 添加一个维度，表示batch size为1

        # input_ = ttf.to_tensor(inputImage)  # 将图片转为张量
        #
        # return input_
        return inputImage

