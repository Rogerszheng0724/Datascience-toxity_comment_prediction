import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 停用詞與斷詞
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 資料切分與 SMOTE 過採樣
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 評估工具與轉換
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score, roc_curve)
from sklearn.preprocessing import label_binarize

# Word2Vec
from gensim.models import Word2Vec

# 羅吉斯回歸
from sklearn.linear_model import LogisticRegression

# 進度條
from tqdm import tqdm

# 下載 NLTK 必要資源
nltk.download('stopwords')
nltk.download('punkt')

###############################################################################
# 前處理：停用詞移除與斷詞
###############################################################################
def remove_stopwords(text):
    """
    將輸入的文字斷詞並移除停用詞（僅保留英文字母單字）。
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    return filtered_tokens

###############################################################################
# Word2Vec 向量化方法
###############################################################################
def get_word2vec_embeddings(token_lists, w2v_model):
    """
    利用訓練好的 Word2Vec 模型將斷詞結果轉換為向量，
    以每筆文本中所有 token 向量的平均作為句子向量。
    加入進度條顯示處理進度。
    """
    embeddings = []
    vector_dim = w2v_model.vector_size
    for tokens in tqdm(token_lists, desc="Word2Vec Embeddings", leave=False):
        vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if vecs:
            embeddings.append(np.mean(vecs, axis=0))
        else:
            embeddings.append(np.zeros(vector_dim))
    return np.array(embeddings)

###############################################################################
# 主流程：利用 SMOTE + 停用詞刪除 + Word2Vec + 羅吉斯回歸 預測多類別 toxicity (0～6)
###############################################################################
def process_toxicity(df, target_col, w2v_model):
    """
    對指定 target_col (toxicity) 進行處理：
      1. 前處理與向量化（使用 Word2Vec）
      2. 資料切分：70% 訓練、30% 均分為驗證與測試（各 15%）
      3. 使用 SMOTE 進行 oversampling（僅對訓練集）
      4. 利用羅吉斯回歸訓練多類別模型（設定 multi_class='multinomial'）
      5. 預測與評估，並產生分類報告、ROC-AUC、混淆矩陣，以及 micro-average Precision-Recall 與 ROC 曲線
    並顯示各階段所花費的時間。
    """
    print(f"\n===== Processing target: {target_col} =====")
    start_total = time.time()
    
    # 1. 前處理：對 comment_text 進行停用詞刪除
    df['clean_tokens'] = df['comment_text'].apply(remove_stopwords)
    
    # 2. 向量化：利用 Word2Vec 產生句子向量
    start_emb = time.time()
    X = get_word2vec_embeddings(df['clean_tokens'], w2v_model)
    end_emb = time.time()
    print(f"Embedding generation time: {end_emb - start_emb:.2f} seconds")
    
    # 3. 取得目標標籤（多類別：0～6）
    y = df[target_col]
    
    # 4. 資料切分：70% 訓練；30% 均分為驗證與測試（各 15%）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 5. SMOTE 過採樣 - 僅對訓練集
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 6. 模型訓練：使用羅吉斯回歸（設定 multi_class='multinomial'）
    start_train = time.time()
    clf = LogisticRegression(random_state=42, solver='saga', multi_class='multinomial', n_jobs=-1, max_iter=1000)
    clf.fit(X_train_res, y_train_res)
    end_train = time.time()
    print(f"Training time: {end_train - start_train:.2f} seconds")
    
    # 7. 預測與評估 (使用測試集)
    start_pred = time.time()
    y_pred = clf.predict(X_test)
    end_pred = time.time()
    print(f"Prediction time: {end_pred - start_pred:.2f} seconds")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    report['roc_auc'] = roc_auc
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC (ovr): {roc_auc:.4f}")
    
    # 8. 視覺化結果
    # (a) 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(f"{target_col}_confusion_matrix.png")
    plt.close()
    
    # (b) Precision-Recall Curve (micro-average)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = clf.predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_scores.ravel())
    avg_precision = average_precision_score(y_test_bin, y_scores, average="micro")
    plt.figure()
    plt.step(recall, precision, where='post', label=f'Micro-average AP = {avg_precision:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-average Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(f"{target_col}_precision_recall.png")
    plt.close()
    
    # (c) ROC Curve (micro-average)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
    plt.figure()
    plt.plot(fpr, tpr, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{target_col}_roc_curve.png")
    plt.close()
    
    end_total = time.time()
    print(f"Total processing time: {end_total - start_total:.2f} seconds")
    
    return clf, report

if __name__ == "__main__":
    # 1. 修改此處路徑為您的資料集路徑（資料集需包含 comment_text 與 toxicity 欄位）
    file_path = r"C:\Users\Admin\Desktop\datascience\toxicity\train.csv"
    df = pd.read_csv(file_path)
    
    # 2. 設定目標欄位（此處僅預測 toxicity，多類別 0～6）
    target_column = "toxicity"
    
    # 3. 利用全文本資料來訓練一個 Word2Vec 模型
    df['clean_tokens'] = df['comment_text'].apply(remove_stopwords)
    w2v_model = Word2Vec(
        sentences=df['clean_tokens'],
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    
    # 4. 執行 toxicity 流程
    clf, report = process_toxicity(df.copy(), target_column, w2v_model=w2v_model)
    
    # 5. 輸出評估報告
    print(f"\n=== {target_column} Evaluation Report ===")
    print(report)
