from flask import Flask,render_template,request
import torch
from train.NetModel import Net
from PIL import Image
import torchvision.transforms as transforms


app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/relic',methods=["POST","GET"])
def relic():
    image_file = request.files['image']
    input_image_path = 'static/input/input.png'
    image_file.save(input_image_path)
    # 加载模型结构
    net = Net()

    # 加载模型参数
    pthfile = 'model_best.pth'
    net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))  # 如果在 CPU 上运行的话需要指定 map_location=torch.device('cpu')
    net.eval()

    # 读取待处理的图片
    input_image = Image.open(image_file).convert('RGB')

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
    output_image_path = 'static/output/output.png'
    output_image.save(output_image_path)

    return '<img src="static/input/input.png" /><img src="/static/output/output.png" />'

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

if __name__ == '__main__':
    app.run(port=5000)