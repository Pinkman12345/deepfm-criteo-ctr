# DeepFM-CTR: 基于 Criteo 数据集的点击率预测与推荐排序实现

一个面向推荐排序/CTR 预估场景的个人项目，基于 **Criteo 1M 数据集** 完成从数据预处理、baseline 建模到 **DeepFM 主模型训练评估** 的完整实验闭环。项目重点体现了我在 **数据处理、特征工程、CTR 预测、模型对比、实验评估** 等方面的实践能力。

---

## 1. 项目简介

点击率预估（CTR Prediction）是推荐系统、广告系统和搜索排序中的核心任务之一，常用于精排阶段预测某条曝光样本被点击的概率。

本项目基于匿名化的 **Criteo 1M** 数据集，使用：

- **Logistic Regression** 作为 baseline
- **DeepFM** 作为主模型

完成 CTR 预测建模，并使用 **AUC** 和 **LogLoss** 对模型效果进行评估。

---

## 2. 数据集说明

数据文件：

```text
data/raw/Criteo_1M_with_nans.csv

字段组成：

target：点击标签，0 表示未点击，1 表示已点击
intCol_0 ~ intCol_12：13 个匿名化数值特征
catCol_0 ~ catCol_25：26 个匿名化类别特征
数据预处理方式
对数值特征缺失值填充为 0
对数值特征进行标准化
对类别特征缺失值填充为 "missing"
对类别特征进行 Label Encoding
按 8:1:1 划分训练集、验证集、测试集
使用分层抽样保证不同数据集标签分布一致
3. 模型说明
3.1 Logistic Regression（Baseline）

使用逻辑回归作为 CTR 预测 baseline，先验证数据预处理、训练评估与结果落盘链路是否打通，并为后续 DeepFM 提供对照基线。

3.2 DeepFM

DeepFM 由三部分组成：

一阶线性部分：学习特征的一阶贡献
FM 部分：学习 sparse 特征之间的低阶交互
Deep 部分：学习高阶非线性特征组合关系

相较于线性模型，DeepFM 更适合 CTR 场景中的稀疏高维特征建模。

4. 项目结构
deepfm-criteo-ctr/
├── data/
│   ├── raw/
│   │   └── Criteo_1M_with_nans.csv
│   └── processed/
│       ├── train_processed.csv
│       ├── valid_processed.csv
│       └── test_processed.csv
├── outputs/
│   ├── models/
│   │   ├── lr_model.joblib
│   │   └── deepfm_best.pth
│   └── results/
│       ├── lr_metrics.json
│       ├── deepfm_metrics.json
│       └── comparison.csv
├── src/
│   ├── preprocess.py
│   ├── check_processed_data.py
│   ├── metrics.py
│   ├── dataset.py
│   ├── train_lr.py
│   ├── train_deepfm.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── make_comparison.py
│   └── models/
│       └── deepfm.py
├── README.md
├── requirements.txt
└── .gitignore
5. 环境安装

建议使用 Python 3.10+。

创建并激活虚拟环境后，安装依赖：

pip install -r requirements.txt
6. 运行方式
6.1 数据预处理
python src/preprocess.py
6.2 检查预处理结果
python src/check_processed_data.py
6.3 训练 Logistic Regression baseline
python src/train_lr.py
6.4 测试 PyTorch Dataset
python src/test_dataset.py
6.5 测试 DeepFM 模型前向传播
python src/test_model.py
6.6 训练 DeepFM
python src/train_deepfm.py
6.7 生成模型结果对比表
python src/make_comparison.py
7. 实验结果

当前实验结果如下：

Model	Valid AUC	Test AUC	Test LogLoss
Logistic Regression	0.5171	0.5139	0.5703
DeepFM	0.7791	0.7824	0.4649
结果分析
Logistic Regression 作为线性 baseline，对匿名化稀疏类别特征的表达能力有限，整体效果较弱。
DeepFM 同时结合了 FM 的低阶交互建模能力和 DNN 的高阶特征学习能力，在 CTR 预测任务上取得了显著更好的效果。
在当前实验设置下，DeepFM 的测试集 AUC 相较 LR 提升明显，验证了深度排序模型在 CTR 场景中的优势。