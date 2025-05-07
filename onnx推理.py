import os

import numpy as np
import onnxruntime as ort
import cv2
import torch

# 导出onnx地址
onnxpath = os.path.join(
    os.path.dirname(__file__), "pth", "flowers.onnx"
)

# 读取图片
img_path = os.path.join(os.path.dirname(__file__), "images", "chuju.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
img = cv2.resize(img, (32, 32))
img = img.astype(np.float32) / 255.0  # 归一化到 [0,1]
img = np.transpose(img, (2, 0, 1))         
img = np.expand_dims(img, axis=0)         
img_tensor = torch.tensor(img, dtype=torch.float32)

# 加载onnx模型
sess = ort.InferenceSession(onnxpath)
classlabels = ["落新妇", "风铃草", "黑眼菊", "金盏花", "金英花", "康乃馨", "雏菊", "金鸡菊", "蒲公英", "鸢尾花","玫瑰", "向日葵", "郁金香", "睡莲"]
outputs = sess.run(None, {"input": img_tensor.detach().numpy()})
output = classlabels[np.argmax(outputs[0])]
print(output)