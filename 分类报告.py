import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

csvdata = pd.read_csv('result/result1.csv')
# 拿到真实标签
true_label = csvdata["label"].values

# 获取预测标签
predict_label = csvdata.iloc[:, :-2].values

# 预测分类及分数的提取
predict_label_ind = np.argmax(predict_label, axis=1)
predict_label_score = np.max(predict_label, axis=1)

# 根据预测值和真实值生成分类报告
report = classification_report(y_true=true_label, y_pred=predict_label_ind)
print(report)
acc = accuracy_score(y_true=true_label, y_pred=predict_label_ind)
print("准确度：", acc)
# 获取精确度Precision
precision = precision_score(y_true=true_label, y_pred=predict_label_ind, average="macro")
print("精确度：", precision)
#召回率(Recall)
recall = recall_score(y_true=true_label, y_pred=predict_label_ind, average="macro")
print("召回率：", recall)
#### F1分数(F1-Score)
f1 = f1_score(y_true=true_label, y_pred=predict_label_ind, average="macro")
print("F1分数：", f1)
