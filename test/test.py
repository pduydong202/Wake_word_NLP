import pandas as pd

# 1. Đọc file JSONL chứa 1800 câu của bạn vào DataFrame
file_path = "900_cau_lenh_giong_noi_viet_xung_ho_doi_thuong.jsonl"
df = pd.read_json(file_path, lines=True)

print(f"Số lượng ban đầu: {len(df)} câu")

# 2. Xóa bỏ các dòng có cột 'text' giống hệt nhau, chỉ giữ lại 1 bản duy nhất
df_clean = df.drop_duplicates(subset=['text'])

print(f"Số lượng sau khi lọc trùng: {len(df_clean)} câu")

# 3. Lưu lại thành file JSONL sạch để đưa vào huấn luyện
df_clean.to_json("dataset_sach.jsonl", orient="records", lines=True, force_ascii=False)
print("Đã lưu file dữ liệu sạch!")