import re
import matplotlib.pyplot as plt

# 读取文本文件
file_path = "log_info.txt"  # 替换为你的文件路径
with open(file_path, "r") as file:
    lines = file.readlines()

# 提取带有"reference_loss:"的行末位的浮点数
losses = []
pattern = r"loss: ([0-9.]+)"
for line in lines:
    match = re.search(pattern, line)
    if match:
        loss = float(match.group(1).rstrip("."))
        if loss < 100:
            losses.append(loss)
mean_losses = [sum(losses[i*100: (i+1)*100]) / 100 for i in range(len(losses) // 100)]
epochs = list(range(1, len(mean_losses) + 1))
epochs = [ i * 100 for i in range(1, len(mean_losses) + 1)]

# 绘制折线图
plt.plot(epochs, mean_losses)
plt.xlabel("Epoch")
plt.ylabel("Mean Loss in Per 100 Epochs")
plt.title("Diffussion Loss over Epochs")
plt.show()