import sys
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from utils import *
from NetModel import Net
from MyDataset import *

if __name__ == '__main__':  # 只有在 main 中才能开多线程
    EPOCH = 50  # 训练次数
    BATCH_SIZE = 16  # 每批的训练数量
    LEARNING_RATE = 0.001  # 学习率
    loss_list = []  # 损失存储数组
    best_psnr = 0  # 训练最好的峰值信噪比
    best_epoch = 0  # 峰值信噪比最好时的 epoch

    inputPathTrain = '../data/trainInput'  # 训练输入图片路径
    #targetPathTrain = 'D:/PycharmProjects/RelicRepair/trainTarget'  # 训练目标图片路径
    targetPathTrain = '../data/trainTarget'  # 训练目标图片路径
    inputPathTest = '../data/testInput'  # 测试输入图片路径
    targetPathTest = '../data/testTarget'  # 测试目标图片路径

    resultPathTest = '../data/Output/'  # 测试结果图片路径

    myNet = Net()  # 实例化网络
    myNet = myNet.cuda()  # 网络放入GPU中
    criterion = nn.MSELoss().cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=LEARNING_RATE)  # 网络参数优化算法

    # 训练数据
    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain)  # 实例化训练数据集类
    # 可迭代数据加载器加载训练数据
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=6,
                             pin_memory=True)

    # 评估数据
    datasetValue = MyValueDataSet(inputPathTest, targetPathTest)  # 实例化评估数据集类
    valueLoader = DataLoader(dataset=datasetValue, batch_size=16, shuffle=True, drop_last=False, num_workers=6,
                             pin_memory=True)

    # 测试数据
    datasetTest = MyTestDataSet(inputPathTest)  # 实例化测试数据集类
    # 可迭代数据加载器加载测试数据
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=6,
                            pin_memory=True)

    # 开始训练
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists('../model_best.pth'):  # 判断是否预训练
        myNet.load_state_dict(torch.load('../model_best.pth'))

    for epoch in range(EPOCH):
        myNet.train()  # 指定网络模型训练状态
        iters = tqdm(trainLoader, file=sys.stdout)  # 实例化 tqdm，自定义
        epochLoss = 0  # 每次训练的损失
        timeStart = time.time()  # 每次训练开始时间
        for index, (x, y) in enumerate(iters, 0):
            myNet.zero_grad()  # 模型参数梯度置0
            optimizer.zero_grad()  # 同上等效

            input_train, target = Variable(x).cuda(), Variable(y).cuda()  # 转为可求导变量并放入 GPU
            output_train = myNet(input_train)  # 输入网络，得到相应输出

            loss = criterion(output_train, target)  # 计算网络输出与目标输出的损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新网络参数
            epochLoss += loss.item()  # 累计一次训练的损失

            # 自定义进度条前缀
            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch + 1, EPOCH, loss.item()))

        # 评估
        myNet.eval()
        psnr_val_rgb = []
        for index, (x, y) in enumerate(valueLoader, 0):
            input_, target_value = x.cuda(), y.cuda()
            with torch.no_grad():
                output_value = myNet(input_)
            for output_value, target_value in zip(output_value, target_value):
                #自己加的
                output_value_np = output_value.cpu().numpy()  # 将 output_value 转换为 numpy 数组
                target_value_np = target_value.cpu().numpy()  # 将 target_value 转换为 numpy 数组

                psnr_val_rgb.append(psnr(output_value, target_value))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(myNet.state_dict(), '../model_best.pth')

        loss_list.append(epochLoss)  # 插入每次训练的损失值
        torch.save(myNet.state_dict(), '../model.pth')  # 每次训练结束保存模型参数
        timeEnd = time.time()  # 每次训练结束时间
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}.".format(epoch + 1, timeEnd - timeStart, epochLoss))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

    # 测试
    print('--------------------------------------------------------------')
    myNet.load_state_dict(torch.load('../model_best.pth'))  # 加载已经训练好的模型参数
    myNet.eval()  # 指定网络模型测试状态

    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, x in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存
            input_test = x.cuda()  # 放入GPU
            output_test = myNet(input_test)  # 输入网络，得到输出
            save_image(output_test, resultPathTest + str(index + 1).zfill(3) + '.png')  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))

    # 绘制训练时损失曲线
    plt.figure(1)
    x = range(0, EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.plot(x, loss_list, 'r-')
    plt.show()
