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

# 機器學習模型與評估
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score, roc_curve)

# Word2Vec
from gensim.models import Word2Vec

# BERT 與 GPU 支援
import torch
from transformers import BertTokenizer, BertModel

# 進度條
from tqdm import tqdm

# 下載 NLTK 必要資源
nltk.download('stopwords')
nltk.download('punkt')

# 設定裝置 (GPU: CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

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
# 向量化方法
###############################################################################
def get_word2vec_embeddings(token_lists, w2v_model):
    """
    利用訓練好的 Word2Vec 模型將斷詞結果 (token_lists) 轉換為向量，
    每筆文本中所有 token 向量的平均作為句子向量。
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

def get_bert_embeddings(texts, tokenizer, bert_model, max_length=128):
    """
    使用 BERT 將每筆原始文字轉換為句子向量，
    tokenize 後取最後一層隱藏層輸出平均作為向量。
    加入進度條顯示處理進度。
    """
    all_embeddings = []
    bert_model.eval()
    for text in tqdm(texts, desc="BERT Embeddings", leave=False):
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # 將所有 tensor 搬到 GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
            all_embeddings.append(embedding)
    return np.array(all_embeddings)

###############################################################################
# 主流程：對單一目標欄位進行處理（訓練、預測、評估與視覺化）
###############################################################################
def process_target(df, target_col, embedding_type='bert', w2v_model=None, tokenizer=None, bert_model=None):
    """
    對指定目標欄位 target_col 進行處理：
      1. 前處理、向量化
      2. 資料切分：70% 訓練、15% 驗證、15% 測試
      3. 訓練集 SMOTE 過採樣
      4. 模型訓練、預測與評估
      5. 產生混淆矩陣、Precision-Recall 與 ROC 曲線圖
    並顯示各階段所花的時間。
    """
    print(f"\n===== Processing target column: {target_col} with {embedding_type} =====")
    start_total = time.time()

    # 1. 前處理：停用詞處理（對於 Word2Vec 使用 clean_tokens）
    df['clean_tokens'] = df['comment_text'].apply(remove_stopwords)
    
    # 2. 向量化
    start_emb = time.time()
    if embedding_type == 'word2vec':
        if w2v_model is None:
            raise ValueError("請先提供 w2v_model (Word2Vec)！")
        X = get_word2vec_embeddings(df['clean_tokens'], w2v_model)
    elif embedding_type == 'bert':
        if tokenizer is None or bert_model is None:
            raise ValueError("請先提供 tokenizer 與 bert_model！")
        X = get_bert_embeddings(df['comment_text'], tokenizer, bert_model)
    else:
        raise ValueError("embedding_type 僅支援 'word2vec' 或 'bert'")
    end_emb = time.time()
    print(f"Embedding generation time: {end_emb - start_emb:.2f} seconds")
    
    # 3. 取得目標標籤
    y = df[target_col]
    
    # 4. 資料切分：70% 訓練，剩下 30% 均分為驗證與測試 (各 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 5. SMOTE 過採樣 - 僅對訓練集
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 6. 訓練 Logistic Regression 模型
    start_train = time.time()
    clf = LogisticRegression(random_state=42, solver='saga', n_jobs=-1, max_iter=1000)
    clf.fit(X_train_res, y_train_res)
    end_train = time.time()
    print(f"Training time: {end_train - start_train:.2f} seconds")
    
    # 7. 預測與評估 (以測試集為例)
    start_pred = time.time()
    y_pred = clf.predict(X_test)
    end_pred = time.time()
    print(f"Prediction time: {end_pred - start_pred:.2f} seconds")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    report['roc_auc'] = roc_auc
    print(f"Classification Report for {target_col}:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # 8. 視覺化
    # (a) 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {target_col} ({embedding_type})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    cm_filename = f"{target_col}_{embedding_type}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    
    # (b) Precision-Recall Curve
    y_scores = clf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    plt.figure()
    plt.step(recall, precision, where='post', label=f'AP = {avg_precision:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {target_col} ({embedding_type})")
    plt.legend(loc="lower left")
    pr_filename = f"{target_col}_{embedding_type}_precision_recall.png"
    plt.savefig(pr_filename)
    plt.close()
    
    # (c) ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {target_col} ({embedding_type})')
    plt.legend(loc="lower right")
    roc_filename = f"{target_col}_{embedding_type}_roc_curve.png"
    plt.savefig(roc_filename)
    plt.close()
    
    end_total = time.time()
    print(f"Total processing time for {target_col}: {end_total - start_total:.2f} seconds")
    
    return clf, report

###############################################################################
# 主程式入口
###############################################################################
if __name__ == "__main__":
    # 1. 請將此處檔案路徑修改為您的測試集檔案路徑
    file_path = r"C:\Users\Admin\Desktop\datascience\bert_dataset\train.csv"
    df = pd.read_csv(file_path)
    
    # 2. 指定六個目標欄位 (預測 0/1)
    target_columns = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']
    
    # 3. 設定使用的向量化方法：'bert' 或 'word2vec'
    use_embedding = 'bert'
    w2v_model = None
    if use_embedding == 'word2vec':
        # 若使用 Word2Vec，先以全資料集訓練一個簡單的模型
        df['clean_tokens'] = df['comment_text'].apply(remove_stopwords)
        w2v_model = Word2Vec(
            sentences=df['clean_tokens'],
            vector_size=100,
            window=5,
            min_count=1,
            workers=4
        )
    
    # 4. 若使用 BERT，請確保模型搬到 GPU
    if use_embedding == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        bert_model.eval()
    
    # 5. 使用進度條依序處理每個目標欄位
    results = {}
    for target in tqdm(target_columns, desc="Processing Targets"):
        if use_embedding == 'word2vec':
            clf, report = process_target(df.copy(), target, embedding_type='word2vec', w2v_model=w2v_model)
        elif use_embedding == 'bert':
            clf, report = process_target(df.copy(), target, embedding_type='bert', tokenizer=tokenizer, bert_model=bert_model)
        results[target] = report
    
    # 6. 輸出所有目標欄位的評估報告
    for target, rep in results.items():
        print(f"\n=== {target} ===")
        print(rep)
