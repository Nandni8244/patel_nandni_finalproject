# patel_nandni_finalproject

**Name:** Nandni Patel  
**Email:** np888@njit.edu  
**Instructor:** Dr. Yasser Abduallah  
**Course:** CS634 – Data Mining

## 1. Overview
This project applies machine learning and deep learning techniques to predict passenger survival in the Estonia Ferry Disaster (1994) based on demographic and categorical information.
Three core algorithms were implemented:

- Random Forest (Ensemble Model – Mandatory)
- LSTM (Deep Learning Model – Mandatory)
- SVM (Classical ML Model – Mandatory)
- (KNN was used only for comparison)

A full 10-fold cross-validation evaluation was performed with detailed metrics and visualizations.

## 2. Dataset
- Name: Passenger List for the Estonia Ferry Disaster
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/christianlillelund/passenger-list-for-the-estonia-ferry-disaster
- Rows: 989
- Features: 8
- Target: Survived (0 = No, 1 = Yes)

**Preprocessing Steps:**
- Removed irrelevant columns
- Label encoding for categorical features
- Standard scaling for numerical features
- Data reshaping for LSTM (3D input)
- Checked and handled missing values

## 3. Algorithms Used
Random Forest
- Handles nonlinear relationships well
- Reduces overfitting
- Works effectively on tabular data

LSTM
- Satisfies the deep learning model requirement
- Learns complex feature interactions
- Input scaled and reshaped to 3D

SVM
- Strong classical baseline algorithm
- Works well on small to medium datasets
- Performance improved with feature scaling

(KNN used only as an additional comparison model.)

## 4. Implementation
- Language: Python
- Environment: Jupyter Notebook + Python scripts
- Evaluation Method: 10-fold cross-validation
- Metrics calculated per fold and overall: TP, TN, FP, FN, Accuracy, Precision, Recall, F1, Specificity, FPR, FNR, Balanced Accuracy, TSS, HSS, ROC, AUC, Brier Score (BS), Brier Skill Score (BSS)

## 5. Installation
Use the following command to install the required libraries:
<br>pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn

<br>**Option 1:** Google Collab
- Open the notebook file.  
- Run each cell sequentially (`Shift + Enter`) to execute and view results.

<br>**Option 2:** Visual Studio Code
- Open the project folder.  
- Run the Python script. For example: python patel_nandni_finalproject.py

## 6. Conclusion
LSTM achieved the highest overall performance, outperforming Random Forest and SVM in predicting survival on the Estonia ferry disaster dataset. The results show that deep learning models can capture complex feature relationships better than traditional methods. Proper preprocessing and consistent evaluation were key to ensuring fair and reliable comparison across all algorithms.
