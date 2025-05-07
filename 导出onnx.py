import os
import torch
from flowers预测 import resnet18_model

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    # 模型地址
    # 导出onnx地址
    onnxpath = os.path.join(
        os.path.dirname(__file__), "pth", "flowers.onnx"
    )
    model_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'weights', 'my_flowers_resnet18.pth'))
    weight_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'weights', 'resnet18.pth'))
    net = resnet18_model(weight_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path))
    # print(net.state_dict())
    net.to(device)
    # 创建一个实例输入
    x = torch.randn(1, 3, 32, 32, device=device)
    # 导出onnx
    torch.onnx.export(
        net,
        x,
        onnxpath,
        verbose=False, # 输出转换过程
        input_names=["input"],
        output_names=["output"],
    )
    print("onnx导出成功")