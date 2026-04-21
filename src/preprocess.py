import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# 1. 基础配置
# =========================
RAW_DATA_PATH = "data/raw/Criteo_1M_with_nans.csv"
PROCESSED_DIR = "data/processed"

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]


def main():
    # 创建输出目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("开始读取原始数据...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"原始数据形状: {df.shape}")

    # =========================
    # 2. 处理数值特征
    # =========================
    print("处理数值特征...")
    df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0)

    scaler = StandardScaler()
    df[DENSE_FEATURES] = scaler.fit_transform(df[DENSE_FEATURES])

    # =========================
    # 3. 处理类别特征
    # =========================
    print("处理类别特征...")
    df[SPARSE_FEATURES] = df[SPARSE_FEATURES].fillna("missing").astype(str)

    for col in SPARSE_FEATURES:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

    # =========================
    # 4. 标签列处理
    # =========================
    print("处理标签列...")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # =========================
    # 5. 划分数据集
    # train : valid : test = 8 : 1 : 1
    # =========================
    print("划分训练/验证/测试集...")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[TARGET_COL]
    )

    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {valid_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    # =========================
    # 6. 保存结果
    # =========================
    train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    valid_path = os.path.join(PROCESSED_DIR, "valid_processed.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("预处理完成，文件已保存：")
    print(train_path)
    print(valid_path)
    print(test_path)


if __name__ == "__main__":
    main()