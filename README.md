# A Review of Imbalanced Data Processing Methods: A Case Study on Toxicity Detection in Social Media

## 📖 專案簡介
本專案探討 **不平衡資料 (Imbalanced Data)** 的處理方法，並以 **社群媒體毒性偵測 (Toxicity Detection in Social Media)** 作為案例研究。透過實驗與比較，說明在不同採樣策略與分類器下的效果。

---

## 🎯 研究目標與貢獻
- 探索 **資料層級 (Data-level)** 與 **模型層級 (Model-level)** 的不平衡資料處理方法  
- 以 **社群媒體毒性檢測** 為例，進行案例實驗  
- 提供 **多種實驗比較** 與 **評估指標** 分析  


---

## 🔄 實驗流程
本研究的實驗流程如下：

![flowchart](https://ppt.cc/fzu4Ax@.png)

---

## 📊 資料集介紹
- 來源：社群媒體毒性留言資料  
- 標籤種類：
  - **identity_hate** (不當仇恨)  
  - **severe_toxic** (嚴重毒性)  
  - **toxic** (毒性)  
  - **threat** (威脅)  
  - **insult** (侮辱)  
  - **obscene** (猥褻)

![flowchart](https://ppt.cc/fhRBKx@.png)
![flowchart](https://ppt.cc/fMzdnx@.png)

---

## ⚙️ 採樣方法
1. **Oversampling** – 增加少數類樣本數量  

2. **Undersampling** – 減少多數類樣本數量  

3. **SMOTE** – 合成少數類樣本 (Synthetic Minority Oversampling Technique)  

---

## 🤖 使用的分類模型
- Logistic Regression
- Random Forest
- XGBoost
- BERT

---

## 🧪 實驗結果
針對不同類別（identity_hate、insult、obscene、severe_toxic、threat、toxic），在不同 **採樣方法 + 分類器** 下進行比較，並以 **Precision、Recall、F1-score、AUC ROC** 作為主要指標。
![flowchart](https://ppt.cc/fVJYnx@.png)
![flowchart](https://ppt.cc/fuMLHx@.png)
![flowchart](https://ppt.cc/fHPXbx@.png)
![flowchart](https://ppt.cc/f0WTxx@.png)
![flowchart](https://ppt.cc/fFFWzx@.png)
![flowchart](https://ppt.cc/f6vvcx@.png)
![flowchart](https://ppt.cc/fiStYx@.png)

---

## 📌 研究結論
- **採樣方法**：SMOTE 在大多數情況下能改善模型對少數類別的學習效果。  
- **分類模型**：BERT 與 XGBoost 在多數任務上表現優於傳統方法。  
- **綜合觀察**：不同類別的最佳組合不一致，需要依任務調整策略。  

【圖片】結果彙整（第27–29頁）

---

## 📚 參考文獻
本研究引用多篇國際期刊與會議文獻，涵蓋 **不平衡資料處理、社群媒體毒性偵測、深度學習與 NLP** 等領域。


[1] A. Sheth, V. L. Shalin, and U. Kursuncu, “Defining and detecting toxicity on social media: context and knowledge are key,” Neurocomputing, vol. 490, pp. 312–318, Jun. 2022, doi: 10.1016/j.neucom.2021.11.095.

[2] Y. Zhang, V. Hangya, and A. Fraser, “A Study of the Class Imbalance Problem in Abusive Language Detection,” in Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024), Y.-L. Chung, Z. Talat, D. Nozza, F. M. Plaza-del-Arco, P. Röttger, A. Mostafazadeh Davani, and A. Calabrese, Eds., Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 38–51. doi: 1
0.18653/v1/2024.woah-1.4.

[3] S. Henning, W. Beluch, A. Fraser, and A. Friedrich, “A Survey of Methods for Addressing Class Imbalance in Deep-Learning Based Natural Language Processing,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, A. Vlachos and I. Augenstein, Eds., Dubrovnik, Croatia: Association for Computational Linguistics, May 2023, pp. 523–540. doi: 10.18653/v1/2023.eacl-main.38.

[4] D. Chatzakou et al., “Detecting Cyberbullying and Cyberaggression in Social Media,” ACM Trans Web, vol. 13, no. 3, p. 17:1-17:51, Oct. 2019, doi: 10.1145/3343484.

[5] “An Image is Worth a Thousand Toxic Words: A Metamorphic Testing Framework for Content Moderation Software | IEEE Conference Publication | IEEE Xplore.” Accessed: Mar. 08, 2025. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/10298324

[6] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.

[7] W. Yin, J. Hay, and D. Roth, “Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Jan. 2019, pp. 3914–3923. doi: 10.18653/v1/D19-1404.

[8] “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets | PLOS One.” Accessed: Mar. 08, 2025. [Online]. Available: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

[9] Y. Yang, K. Zha, Y. Chen, H. Wang, and D. Katabi, “Delving into Deep Imbalanced Regression,” in Proceedings of the 38th International Conference on Machine Learning, PMLR, Jul. 2021, pp. 11842–11851. Accessed: Mar. 09, 2025. [Online]. Available: https://proceedings.mlr.press/v139/yang21m.html

[10] I. M. Alkhawaldeh, I. Albalkhi, and A. J. Naswhan, “Challenges and limitations of synthetic minority oversampling techniques in machine learning,” World J. Methodol., vol. 13, no. 5, pp. 373–378, Dec. 2023, doi: 10.5662/wjm.v13.i5.373.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 42, no. 2, pp. 318–327, Feb. 2020, doi: 10.1109/TPAMI.2018.2858826.

[12] A. Gupta and S. Gupta, “Enhanced Classification of Imbalanced Medical Datasets using Hybrid Data-Level, Cost-Sensitive and Ensemble Methods,” Int. Res. J. Multidiscip. Technovation, pp. 58–76, Apr. 2024, doi: 10.54392/irjmt2435.

[13] “Exploratory Undersampling for Class-Imbalance Learning | IEEE Journals & Magazine | IEEE Xplore.” Accessed: Mar. 09, 2025. [Online]. Available: https://ieeexplore.ieee.org/document/4717268

[14] “How to Deal With Imbalanced Classification and Regression Data.” Accessed: Mar. 09, 2025. [Online]. Available: https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data

[15] “Mean values of the g-mean and the accuracy of the classifier | Download Table.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.researchgate.net/figure/Mean-values-of-the-g-mean-and-the-accuracy-of-the-classifier_tbl2_221582715

[16] “Class-Balanced Loss Based on Effective Number of Samples | IEEE Conference Publication | IEEE Xplore.” Accessed: Mar. 09, 2025. [Online]. Available: https://ieeexplore.ieee.org/document/8953804?denied=

[17] “[PDF] Learning to Reweight Examples for Robust Deep Learning | Semantic Scholar.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.semanticscholar.org/paper/Learning-to-Reweight-Examples-for-Robust-Deep-Ren-Zeng/c5420ef59d7508d82e53671b0d623027eb58e6ed

[18] S.-K. Hung and J. Q. Gan, “Augmentation of Small Training Data Using GANs for Enhancing the Performance of Image Classification,” in 2020 25th International Conference on Pattern Recognition (ICPR), Jan. 2021, pp. 3350–3356. doi: 10.1109/ICPR48806.2021.9412399.

[19] S. Motamed, P. Rogalla, and F. Khalvati, “Data augmentation using Generative Adversarial Networks (GANs) for GAN-based detection of Pneumonia and COVID-19 in chest X-ray images,” Inform. Med. Unlocked, vol. 27, p. 100779, Jan. 2021, doi: 10.1016/j.imu.2021.100779.

[20] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human 
Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds., Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186. doi: 10.18653/v1/N19-1423.

[21] L. Breiman, “Random Forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001, doi: 10.1023/A:1010933404324.

[22] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Aug. 2016, pp. 785–794. doi: 10.1145/2939672.2939785.

[23] J. Zhao, J. Jin, S. Chen, R. Zhang, B. Yu, and Q. Liu, “A weighted hybrid ensemble method for classifying imbalanced data,” Knowl.-Based Syst., vol. 203, p. 106087, Sep. 2020, doi: 10.1016/j.knosys.2020.106087.

[24] “[PDF] RUSBoost: Improving classification performance when training data is skewed | Semantic Scholar.” Accessed: Mar. 09, 2025. [Online]. Available: https://www.semanticscholar.org/paper/RUSBoost%3A-Improving-classification-performance-when-Seiffert-
Khoshgoftaar/7fb25f0e9eb682bb4cb5ed09e9bc36acc153b88a

[25] D. Bamman, B. O’Connor, and N. Smith, “Censorship and deletion practices in Chinese social media,” First Monday, Mar. 2012, doi: 10.5210/fm.v17i3.3943.

[26] A. P. Bradley, “The use of the area under the ROC curve in the evaluation of machine learning algorithms,” Pattern Recognit., vol. 30, no. 7, pp. 1145–1159, Jul. 1997, doi: 10.1016/S0031-3203(96)00142-2.

[27] T. Fawcett, “An introduction to ROC analysis,” Pattern Recognit. Lett., vol. 27, no. 8, pp. 861–874, Jun. 2006, doi: 10.1016/j.patrec.2005.10.010.

[28] “Assessing Censorship on Microblogs in China: Discriminatory Keyword Analysis and the Real-Name Registration Policy | IEEE Journals & Magazine | IEEE Xplore.” Accessed: Mar. 09, 2025. [Online]. Available: https://ieeexplore.ieee.org/document/6459495

[29] M. Beck et al., “xLSTM: Extended Long Short-Term Memory,” Dec. 06, 2024, arXiv: arXiv:2405.04517. doi: 10.48550/arXiv.2405.04517.


