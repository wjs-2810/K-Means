# 基于 K-Means 的图像矢量量化

Python 机器学习课程实验：手写 K-Means 聚类算法，并将其应用于彩色图像的矢量量化（有损压缩）。

## 实验内容

1. **数据准备**：读取彩色图片 `data/flower.jpg`，将像素值缩放到 0~1，每个像素转换为一条 `[R, G, B]` 三维样本（全部像素参与训练，不抽样）。
2. **手写 K-Means**：完整实现"初始化中心 → 分配样本 → 更新中心 → 判断收敛"的迭代过程。
3. **肘击法选 K**：K 从 1 扫描到 16，每个 K 换 3 个随机种子取最小 SSE，绘制 SSE-K 曲线，用 Kneedle 方法自动定位肘部，确定最优 K。
4. **矢量量化重建**：用 K=2、最优 K、K=30 分别重建图像，将每个像素替换为所属簇的代表颜色，对比压缩效果与压缩率。

## 环境要求

- Python 3.8+
- numpy
- Pillow
- matplotlib

```bash
pip install -r requirements.txt
```

## 运行方式

```bash
python3 kmeans_image_quantization.py
```

## 输出结果

运行后在 `results/` 目录生成：

| 文件 | 说明 |
| --- | --- |
| `original_image.png` | 原图 |
| `elbow_curve.png` | 肘击法 SSE-K 曲线 |
| `reconstruction_comparison.png` | 原图与 K=2、最优 K、K=30 重建图对比 |

## 目录结构

```
.
├── kmeans_image_quantization.py   # 主程序
├── requirements.txt               # 依赖
├── data/
│   └── flower.jpg                 # 实验数据（640×427 彩色照片）
└── results/                       # 输出结果
```

## 实验结论

- 肘击法确定该图片的最优 K = 4（SSE 在 K=4 之后下降明显变缓）。
- K 越大，代表颜色越多，重建图像越接近原图，但压缩率越低：
  - K=2：每像素 1 bit，压缩比约 24:1，细节严重丢失；
  - K=4：每像素 2 bit，压缩比约 12:1，保留主要层次；
  - K=30：每像素约 4.9 bit，压缩比约 4.9:1，接近原图。
