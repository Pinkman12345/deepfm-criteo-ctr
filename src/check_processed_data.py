import pandas as pd

TARGET_COL = "target"
DENSE_FEATURES = [f"intCol_{i}" for i in range(13)]
SPARSE_FEATURES = [f"catCol_{i}" for i in range(26)]

TRAIN_PATH = "data/processed/train_processed.csv"
VALID_PATH = "data/processed/valid_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"


def check_file(path, name):
    print(f"\n========== 检查 {name} ==========")
    df = pd.read_csv(path)

    print(f"{name} 形状: {df.shape}")
    print(f"{name} 前 3 行:")
    print(df.head(3))

    print(f"\n{name} 缺失值总数: {df.isnull().sum().sum()}")
    print(f"{name} target 分布:")
    print(df[TARGET_COL].value_counts(normalize=True))

    print(f"\n{name} 数值特征示例统计:")
    print(df[DENSE_FEATURES].describe().iloc[:2])  # 只看 count / mean

    print(f"\n{name} 类别特征示例:")
    print(df[SPARSE_FEATURES[:3]].head(3))


def main():
    check_file(TRAIN_PATH, "train")
    check_file(VALID_PATH, "valid")
    check_file(TEST_PATH, "test")


if __name__ == "__main__":
    main()