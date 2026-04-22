import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Model": ["YOLOv8n", "SSD", "Faster R-CNN", "EfficientDet"],
    "mAP50": [0.75, 0.68, 0.82, 0.85],
    "Speed": [60, 40, 15, 25]
}

df = pd.DataFrame(data)
df["Efficiency"] = df["mAP50"] * df["Speed"]

epochs = list(range(1, 51))
yolov8n = [0.50 + i*0.005 for i in epochs]
ssd = [0.48 + i*0.004 for i in epochs]
faster = [0.52 + i*0.006 for i in epochs]
efficient = [0.53 + i*0.0065 for i in epochs]

# -------------------------------
# MULTI GRAPH VIEW
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 1. Efficiency
axs[0, 0].bar(df["Model"], df["Efficiency"])
axs[0, 0].set_title("Efficiency")

# 2. Accuracy
axs[0, 1].bar(df["Model"], df["mAP50"])
axs[0, 1].set_title("Accuracy")

# 3. Scatter
axs[1, 0].scatter(df["Speed"], df["mAP50"])
for i, model in enumerate(df["Model"]):
    axs[1, 0].text(df["Speed"][i], df["mAP50"][i], model)
axs[1, 0].set_title("Speed vs Accuracy")

# 4. Line graph
axs[1, 1].plot(epochs, yolov8n, label="YOLOv8n")
axs[1, 1].plot(epochs, ssd, label="SSD")
axs[1, 1].plot(epochs, faster, label="Faster R-CNN")
axs[1, 1].plot(epochs, efficient, label="EfficientDet")
axs[1, 1].legend()
axs[1, 1].set_title("Training Trend")

plt.tight_layout()
plt.show()