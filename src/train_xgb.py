import os
import json
import pandas as pd
from xgboost import XGBClassifier
from joblib import dump

from metrics import compute_auc, compute_logloss

TRAIN_PATH = "data/processed/train_processed.csv"
VALID_PATH = "data/processed/valid_processed.csv"
TEST_PATH = "data/processed/test_processed.csv"

OUTPUT_MODEL_DIR = "outputs/models"
OUTPUT_RESULT_DIR = "outputs/results"

TARGET_COL = "target"
FEATURE_COLS = [f"intCol_{i}" for i in range(13)] + [f"catCol_{i}" for i in range(26)]


def main():
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)

    print("读取训练/验证/测试数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print("开始训练 XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("在验证集上评估...")
    valid_pred = model.predict_proba(X_valid)[:, 1]
    valid_auc = compute_auc(y_valid, valid_pred)
    valid_logloss = compute_logloss(y_valid, valid_pred)

    print("在测试集上评估...")
    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = compute_auc(y_test, test_pred)
    test_logloss = compute_logloss(y_test, test_pred)

    print(f"Valid AUC: {valid_auc:.6f}")
    print(f"Valid LogLoss: {valid_logloss:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(f"Test LogLoss: {test_logloss:.6f}")

    model_path = os.path.join(OUTPUT_MODEL_DIR, "xgb_model.joblib")
    result_path = os.path.join(OUTPUT_RESULT_DIR, "xgb_metrics.json")

    dump(model, model_path)

    results = {
        "model": "XGBoost",
        "valid_auc": valid_auc,
        "valid_logloss": valid_logloss,
        "test_auc": test_auc,
        "test_logloss": test_logloss
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("模型已保存到：", model_path)
    print("评估结果已保存到：", result_path)


if __name__ == "__main__":
    main()