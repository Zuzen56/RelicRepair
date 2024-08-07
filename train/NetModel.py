import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inconv = nn.Sequential(  # 输入层网络
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.midconv = nn.Sequential(  # 中间层网络

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Sequential(  # 输出层网络
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.inconv(x)
        x = self.midconv(x)
        x = self.outconv(x)

        return x
