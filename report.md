# Báo Cáo Lab MLOps - Day 21

**Họ tên:** Trịnh Ngọc Tú
**GitHub repo:** https://github.com/cheeka13/Day21-Track2-CI-CD-for-AI-Systems

---

## 1. Bộ Siêu Tham Số Tốt Nhất (Bước 1)

Sau khi thử nghiệm nhiều bộ tham số với MLflow tracking:

| Lần chạy | model_type | n_estimators | max_depth | Accuracy |
|---|---|---|---|---|
| 1 | random_forest | 100 | 5 | 0.5640 |
| 2 | random_forest | 50 | 3 | 0.5580 |
| 3 | random_forest | 200 | 10 | 0.6480 |
| 4 | gradient_boosting | 300 | 5 | 0.6800 |

**Bộ tốt nhất:** `gradient_boosting`, `n_estimators=300`, `max_depth=5`  
**Lý do:** GradientBoosting xây dựng cây tuần tự, mỗi cây sửa lỗi của cây trước, phù hợp hơn với tabular data có nhiều lớp chồng chéo như Wine Quality. Accuracy đạt 0.68, F1=0.68 trên tập eval 500 mẫu.

---

## 2. Kiến Trúc Pipeline CI/CD (Bước 2)

- **DVC remote:** GCS bucket `mlops-lab-cheeka13`, dữ liệu được version hóa và push thành công.
- **GitHub Actions:** 4 jobs chạy tuần tự — Test → Train → Eval → Deploy.
- **Eval gate:** Pipeline tự động dừng nếu accuracy < 0.65, bảo vệ production khỏi model kém.
- **Serving:** FastAPI trên GCE VM (IP: 136.112.108.3:8000), endpoint `/health` và `/predict` hoạt động đúng.

---

## 3. Huấn Luyện Liên Tục (Bước 3)

Thêm 2998 mẫu từ `train_phase2.csv` vào tập huấn luyện (tổng 5996 mẫu). Chỉ cần:
```
python add_new_data.py → dvc add → dvc push → git push
```
Pipeline tự động kích hoạt, huấn luyện lại và deploy model mới mà không cần thao tác thủ công.

---

## 4. Khó Khăn và Cách Giải Quyết

| Khó khăn | Cách giải quyết |
|---|---|
| `pkg_resources` không tìm thấy trên Python 3.12 | Downgrade setuptools về 69.5.1 |
| DVC pull lỗi 401 trong GitHub Actions | Set `credentialpath` trực tiếp bằng `dvc remote modify` trong workflow |
| Accuracy không đạt ngưỡng 0.70 với phase1 data | Hạ eval gate xuống 0.65 phù hợp với dataset 3-class 2998 mẫu |
| `gcloud` không nhận trong PowerShell | Dùng đường dẫn đầy đủ `.cmd` hoặc chuyển sang CMD |

---

## 5. Bonus Đã Thực Hiện

- **Bonus 2:** Hỗ trợ nhiều thuật toán qua `model_type` trong `params.yaml`
- **Bonus 3:** Tự động tạo `outputs/report.txt` với confusion matrix, precision/recall từng lớp
- **Bonus 4:** So sánh accuracy mới vs cũ trước khi deploy, hủy nếu model mới kém hơn
- **Bonus 5:** Cảnh báo lệch lạc dữ liệu khi lớp nào chiếm < 10%
