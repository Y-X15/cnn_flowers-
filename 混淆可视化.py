import os
from sklearn.metrics import *
import pandas as pd
from matplotlib import pyplot as plt

# plt中文乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

current_path = os.path.dirname(__file__)
csv_path = os.path.relpath(
    os.path.join(current_path, r"./result", "result1.csv")
)


def report():
    excel_data = pd.read_csv(csv_path)
    label = excel_data["label"].values
    predict = excel_data["predict"].values
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    matrix = confusion_matrix(label, predict)
    print(matrix)
    plt.matshow(matrix, cmap=plt.cm.Greens)
    # 显示颜色条
    plt.colorbar()
    # 显示具体的数字的过程
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.annotate(
                matrix[i, j],
                xy=(j, i),
                horizontalalignment="center",
                verticalalignment="center",
            )
    plt.xlabel("Pred labels")
    plt.ylabel("True labels")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title("训练结果混淆矩阵视图")

    plt.show()


if __name__ == "__main__":
    report()
