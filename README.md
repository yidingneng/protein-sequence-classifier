Protein-MDFA-Classifier
📌 Overview
This project provides a pipeline for protein sequence feature extraction, classification, and 5-fold evaluation using an MDFA-enhanced ensemble model.
## 📁 Project Structure
├── extract.py      # Feature extraction: AAC / dipeptide / PseAAC / CTD / ProtBert
├── predict.py      # Single train-test run with MDFA + SVM + GBDT ensemble
├── predict_5.py    # 5-fold cross-validation with full metrics & plots
│
└── data/           # Input data (raw sequences and/or extracted feature CSVs)

📦 Requirements 
Python 3.8+
pip install torch numpy pandas scikit-learn transformers biopython matplotlib seaborn tqdm joblib
🧭 Workflow Diagram 
flowchart TD

A[Raw sequences<br>(sequence,+label)] --> B[extract.py<br>Feature extraction]
B --> C[Feature CSV]

C --> D[predict.py<br>Single train/test ensemble]
C --> E[predict_5.py<br>5-fold cross-validation]

D --> F[Reports, confusion matrix,<br>saved model]
E --> G[Cross-validation metrics,<br>ROC, metrics tables]
📌 Notes

You may need to adjust file paths (filepath, RESULTS_PATH, etc.) to match your own directory structure.

If you want to deploy the model or use it on new data, reuse the saved *.pkl model and scaler.
