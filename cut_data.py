import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 讀取資料
df = pd.read_csv(r"C:\Users\Admin\Desktop\datascience\origin\train_preprocessed.csv")


# 2. 先切分出 70% 作為訓練集，剩下 30% 暫存
train, temp = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)

# 3. 將剩下的 30% 再均分為 15% 驗證集與 15% 測試集
val, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)

# 4. 定義輸出資料夾路徑
output_folder = r"C:\Users\Admin\Desktop\datascience\bert_dataset"

# 若資料夾不存在，則建立它
os.makedirs(output_folder, exist_ok=True)

# 5. 輸出成 CSV 檔
train.to_csv(os.path.join(output_folder, "train.csv"), index=False)
val.to_csv(os.path.join(output_folder, "val.csv"), index=False)
test.to_csv(os.path.join(output_folder, "test.csv"), index=False)

# 印出各資料集大小以供檢查
print(f"Train size: {len(train)}")      # 約 70%
print(f"Validation size: {len(val)}")   # 約 15%
print(f"Test size: {len(test)}")        # 約 15%
