import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

EVAL_THRESHOLD = 0.70

LABEL_MAP = {0: "thấp", 1: "trung_bình", 2: "cao"}


def _check_label_distribution(y, split_name: str) -> dict:
    """
    Bonus 5: Kiểm tra phân phối nhãn và cảnh báo nếu lớp nào < 10%.

    Trả về dict tỷ lệ phân phối nhãn.
    """
    total = len(y)
    distribution = {}
    for label in sorted(y.unique()):
        ratio = (y == label).sum() / total
        distribution[f"label_{label}_ratio"] = round(ratio, 4)
        if ratio < 0.10:
            print(
                f"[CẢNH BÁO] {split_name}: Lớp {label} chỉ chiếm "
                f"{ratio:.1%} tổng mẫu (< 10%). Có thể bị mất cân bằng dữ liệu."
            )
    return distribution


def _build_model(params: dict):
    """
    Bonus 2: Chọn thuật toán dựa trên tham số model_type.

    Hỗ trợ: random_forest (mặc định), gradient_boosting, logistic_regression.
    """
    model_type = params.get("model_type", "random_forest")
    # Loại bỏ model_type khỏi params trước khi truyền vào constructor
    rf_params = {k: v for k, v in params.items() if k != "model_type"}

    if model_type == "gradient_boosting":
        allowed = {"n_estimators", "max_depth", "min_samples_split"}
        gb_params = {k: v for k, v in rf_params.items() if k in allowed}
        return GradientBoostingClassifier(**gb_params, random_state=42)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    else:
        # random_forest (mặc định)
        return RandomForestClassifier(**rf_params, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params    : dict chứa các siêu tham số cho mô hình.
        data_path : đường dẫn đến file dữ liệu huấn luyện.
        eval_path : đường dẫn đến file dữ liệu đánh giá.

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá.
    """

    # 1. Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # Bonus 5: Kiểm tra phân phối nhãn trước khi huấn luyện
    label_dist = _check_label_distribution(y_train, "train")

    with mlflow.start_run():

        # 3. Ghi nhận siêu tham số
        mlflow.log_params(params)

        # 4. Khởi tạo và huấn luyện mô hình (Bonus 2: hỗ trợ nhiều thuật toán)
        model = _build_model(params)
        model.fit(X_train, y_train)

        # 5. Dự đoán và tính chỉ số
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # Bonus 3: Tính precision và recall theo từng lớp
        precision_per_class = precision_score(
            y_eval, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        recall_per_class = recall_score(
            y_eval, preds, average=None, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])

        # 6. Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        # Bonus 3: log thêm precision/recall từng lớp
        for i, label in enumerate([0, 1, 2]):
            mlflow.log_metric(f"precision_class_{label}", precision_per_class[i])
            mlflow.log_metric(f"recall_class_{label}", recall_per_class[i])
        # Bonus 5: log phân phối nhãn
        for k, v in label_dist.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model")

        # 7. In kết quả ra màn hình
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # 8. Lưu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
        }
        # Bonus 3: thêm precision/recall từng lớp vào metrics.json
        for i, label in enumerate([0, 1, 2]):
            metrics[f"precision_class_{label}"] = float(precision_per_class[i])
            metrics[f"recall_class_{label}"] = float(recall_per_class[i])
        # Bonus 5: thêm phân phối nhãn vào metrics.json
        metrics.update(label_dist)

        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Bonus 3: Tạo báo cáo hiệu suất dạng văn bản
        _write_report(acc, f1, precision_per_class, recall_per_class, cm, label_dist)

        # 9. Lưu mô hình ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


def _write_report(acc, f1, precision_per_class, recall_per_class, cm, label_dist):
    """Bonus 3: Ghi báo cáo hiệu suất ra outputs/report.txt."""
    os.makedirs("outputs", exist_ok=True)
    lines = []
    lines.append("=" * 50)
    lines.append("BÁO CÁO HIỆU SUẤT MÔ HÌNH")
    lines.append("=" * 50)
    lines.append(f"Accuracy : {acc:.4f}")
    lines.append(f"F1 Score : {f1:.4f}")
    lines.append("")
    lines.append("Precision và Recall theo từng lớp:")
    lines.append(f"  Lớp 0 (thấp)       - Precision: {precision_per_class[0]:.4f} | Recall: {recall_per_class[0]:.4f}")
    lines.append(f"  Lớp 1 (trung_bình) - Precision: {precision_per_class[1]:.4f} | Recall: {recall_per_class[1]:.4f}")
    lines.append(f"  Lớp 2 (cao)        - Precision: {precision_per_class[2]:.4f} | Recall: {recall_per_class[2]:.4f}")
    lines.append("")
    lines.append("Confusion Matrix (hàng=thực tế, cột=dự đoán):")
    lines.append("       Pred_0  Pred_1  Pred_2")
    for i, row in enumerate(cm):
        lines.append(f"  Act_{i}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    lines.append("")
    lines.append("Phân phối nhãn trong tập huấn luyện:")
    for k, v in label_dist.items():
        lines.append(f"  {k}: {v:.4f} ({v:.1%})")
    lines.append("=" * 50)

    report_text = "\n".join(lines)
    print(report_text)
    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)
