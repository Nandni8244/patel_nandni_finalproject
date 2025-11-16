import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import brier_score_loss, auc

from keras.models import Sequential
from keras.layers import Dense, LSTM

data = pd.read_csv('estonia_passenger_list.csv')
data.describe()

data.info()

def impute_missing_values(df):
    for col in df.columns:

        if df[col].dtype != 'object':

            df[col].fillna(df[col].median(), inplace=True)

        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

data = impute_missing_values(data)

data.head()

features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder

data_encoded = data.copy()
categorical_cols = data_encoded.select_dtypes(include=['object']).columns

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data_encoded[col].astype(str))
    label_encoders[col] = le

features = data_encoded.drop('Survived', axis=1)
labels = data_encoded['Survived']

sns.countplot(x=labels, label="Count")
plt.title("Distribution of Survival Outcome")
plt.show()

positive_outcomes, negative_outcomes = labels.value_counts()
total_samples = labels.count()

print("=== Survival Outcome Distribution ===")
print(f"Total Samples           : {total_samples}")
print(f"Survived (1) Count      : {positive_outcomes}")
print(f"Survived (1) Percentage : {round((positive_outcomes / total_samples) * 100, 2)}%")
print(f"Not Survived (0) Count  : {negative_outcomes}")
print(f"Not Survived (0) %      : {round((negative_outcomes / total_samples) * 100, 2)}%")
print("=====================================\n")

numeric_features = features.select_dtypes(include=['int64', 'float64'])

fig, axis = plt.subplots(figsize=(10, 8))
correlation_matrix = numeric_features.corr()

sns.heatmap(correlation_matrix, annot=True, linewidths=.5, fmt='.2f', cmap='coolwarm', ax=axis)
plt.title("Correlation Heatmap")
plt.show()

numeric_features = features.select_dtypes(include=['int64', 'float64'])

numeric_features.hist(figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Features", fontsize=16)
plt.show()

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'Survived' in numeric_cols:
    numeric_cols.remove('Survived')

pairplot_cols = numeric_cols[:4]

sns.pairplot(data[pairplot_cols + ['Survived']], hue='Survived', palette='dark')
plt.show()

features_train_all, features_test_all, labels_train_all, labels_test_all = \
train_test_split(
    features,
    labels,
    test_size=0.1,
    random_state=21,
    stratify=labels
)

for dataset in [features_train_all, features_test_all, labels_train_all, labels_test_all]:
    dataset.reset_index(drop=True, inplace=True)

numeric_cols = features_train_all.select_dtypes(include=['int64', 'float64']).columns

features_train_all_std = features_train_all.copy()
features_train_all_std[numeric_cols] = (
    features_train_all[numeric_cols] - features_train_all[numeric_cols].mean()
) / features_train_all[numeric_cols].std()

features_test_all_std = features_test_all.copy()
features_test_all_std[numeric_cols] = (
    features_test_all[numeric_cols] - features_test_all[numeric_cols].mean()
) / features_test_all[numeric_cols].std()

features_train_all_std.describe()

def calc_metrics(conf_matrix):

    TP = conf_matrix[0][0]
    FN = conf_matrix[0][1]
    FP = conf_matrix[1][0]
    TN = conf_matrix[1][1]

    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    FPR = FP / (TN + FP) if (TN + FP) != 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1_measure = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    BACC = (TPR + TNR) / 2
    TSS = TPR - FPR
    HSS = (
        2 * (TP * TN - FP * FN) /
        ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
        if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) != 0 else 0
    )

    return [
        TP, TN, FP, FN,
        TPR, TNR, FPR, FNR,
        Precision, F1_measure, Accuracy, Error_rate,
        BACC, TSS, HSS
    ]



def get_metrics(model, X_train, X_test, y_train, y_test, LSTM_flag):

    def calc_metrics(conf_matrix):
        TP, FN = conf_matrix[0][0], conf_matrix[0][1]
        FP, TN = conf_matrix[1][0], conf_matrix[1][1]

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FPR = FP / (TN + FP)
        FNR = FN / (TP + FN)
        Precision = TP / (TP + FP)
        F1_measure = 2 * TP / (2 * TP + FP + FN)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        Error_rate = (FP + FN) / (TP + FP + FN + TN)
        BACC = (TPR + TNR) / 2
        TSS = TPR - FPR
        HSS = 2 * (TP * TN - FP * FN) / (
            (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
        )

        return [
            TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure,
            Accuracy, Error_rate, BACC, TSS, HSS
        ]

    metrics = []

    if LSTM_flag == 1:

        model.fit(X_train, y_train, epochs=10, verbose=0)

        y_pred_prob = model.predict(X_test).ravel()

        y_pred = (y_pred_prob > 0.5).astype(int)

        matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])

        brier = brier_score_loss(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics.extend(calc_metrics(matrix))
        metrics.extend([brier, auc, 0])

        return metrics
    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])

        y_pred_prob = model.predict_proba(X_test)[:, 1]

        brier = brier_score_loss(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        acc = model.score(X_test, y_test)

        metrics.extend(calc_metrics(matrix))
        metrics.extend([brier, auc, acc])

        return metrics

knn_parameters = {
    "n_neighbors": list(range(1, 16))
}

knn_model = KNeighborsClassifier()

knn_cv = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_parameters,
    cv=10,
    n_jobs=-1
)

knn_cv.fit(features_train_all_std, labels_train_all)
print("===== KNN GridSearchCV Results =====")
print(f"Best Number of Neighbors (n_neighbors): {knn_cv.best_params_['n_neighbors']}")
print("\n")

best_n_neighbors = knn_cv.best_params_['n_neighbors']
best_n_neighbors

param_grid_rf = {
    "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "min_samples_split": [2, 4, 6, 8, 10]
}

rf_classifier = RandomForestClassifier()

grid_search_rf = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid_rf,
    cv=10,
    n_jobs=-1
)

grid_search_rf.fit(features_train_all_std, labels_train_all)

best_rf_params = grid_search_rf.best_params_
print("===== Random Forest GridSearchCV Results =====")
print(f"Best n_estimators       : {best_rf_params['n_estimators']}")
print(f"Best min_samples_split  : {best_rf_params['min_samples_split']}")
print("\n")


min_samples_split = best_rf_params['min_samples_split']
n_estimators = best_rf_params['n_estimators']

param_grid_svc = {
    "kernel": ["linear"],
    "C": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

svc_classifier = SVC(probability=True)

grid_search_svc = GridSearchCV(
    estimator=svc_classifier,
    param_grid=param_grid_svc,
    cv=10,
    n_jobs=-1
)

grid_search_svc.fit(features_train_all_std, labels_train_all)

best_svc_params = grid_search_svc.best_params_
print("===== Support Vector Machine GridSearchCV Results =====")
print(f"Best Kernel : {best_svc_params['kernel']}")
print(f"Best C Value: {best_svc_params['C']}")
print("\n")


C_value = best_svc_params['C']

cv_stratified = StratifiedKFold(n_splits=10, shuffle=True, random_state=21)

metric_columns = [
    'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
    'Precision', 'F1_measure', 'Accuracy', 'Error_rate',
    'BACC', 'TSS', 'HSS', 'Brier_score', 'AUC', 'Acc_by_package_fn'
]

knn_metrics_list = []
rf_metrics_list = []
svm_metrics_list = []
lstm_metrics_list = []

print("Starting 10-fold cross-validation...\n")

for iter_num, (train_index, test_index) in enumerate(
    cv_stratified.split(features_train_all_std, labels_train_all), start=1):

    features_train = features_train_all_std.iloc[train_index, :]
    features_test = features_train_all_std.iloc[test_index, :]

    labels_train = labels_train_all[train_index]
    labels_test = labels_train_all[test_index]

    knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)

    rf_model = RandomForestClassifier(
        min_samples_split=min_samples_split,
        n_estimators=n_estimators
    )

    svm_model = SVC(C=C_value, kernel='linear', probability=True)

    X_train_lstm = np.array(features_train).reshape(-1, 1, features_train.shape[1])
    X_test_lstm = np.array(features_test).reshape(-1, 1, features_test.shape[1])

    lstm_model = Sequential()
    lstm_model.add(LSTM(
        64,
        activation='relu',
        input_shape=(1, features_train.shape[1])
    ))
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    knn_metrics = get_metrics(knn_model, features_train, features_test, labels_train, labels_test, 0)
    rf_metrics = get_metrics(rf_model, features_train, features_test, labels_train, labels_test, 0)
    svm_metrics = get_metrics(svm_model, features_train, features_test, labels_train, labels_test, 0)
    lstm_metrics = get_metrics(lstm_model, X_train_lstm, X_test_lstm, labels_train, labels_test, 1)

    knn_metrics_list.append(knn_metrics)
    rf_metrics_list.append(rf_metrics)
    svm_metrics_list.append(svm_metrics)
    lstm_metrics_list.append(lstm_metrics)

    metrics_all_df = pd.DataFrame(
        [knn_metrics, rf_metrics, svm_metrics, lstm_metrics],
        columns=metric_columns,
        index=['KNN', 'RF', 'SVM', 'LSTM']
    )

    print(f"\nMetrics in Iteration {iter_num}:-\n")
    print(metrics_all_df.round(decimals=2).T)
    print("\n")

metric_index_df = [
    'iter1', 'iter2', 'iter3', 'iter4', 'iter5',
    'iter6', 'iter7', 'iter8', 'iter9', 'iter10'
]

knn_metrics_df = pd.DataFrame(knn_metrics_list, columns=metric_columns, index=metric_index_df)
rf_metrics_df = pd.DataFrame(rf_metrics_list, columns=metric_columns, index=metric_index_df)
svm_metrics_df = pd.DataFrame(svm_metrics_list, columns=metric_columns, index=metric_index_df)
lstm_metrics_df = pd.DataFrame(lstm_metrics_list, columns=metric_columns, index=metric_index_df)

algorithm_names = ["KNN", "Random Forest", "SVM", "LSTM"]
metrics_dfs = [knn_metrics_df, rf_metrics_df, svm_metrics_df, lstm_metrics_df]

for name, df in zip(algorithm_names, metrics_dfs):
    print(f"\n===== Metrics for {name} =====\n")
    print(df.round(3).T.to_string())

knn_avg_df = knn_metrics_df.mean()
rf_avg_df = rf_metrics_df.mean()
svm_avg_df = svm_metrics_df.mean()
lstm_avg_df = lstm_metrics_df.mean()

avg_performance_df = pd.DataFrame(
    {
        'KNN': knn_avg_df,
        'RF': rf_avg_df,
        'SVM': svm_avg_df,
        'LSTM': lstm_avg_df
    },
    index=metric_columns
)

print("\n===== Average Performance of All Algorithms (10-Fold CV) =====\n")
print(avg_performance_df.round(2))
print("\n")

knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_model.fit(features_train_all_std, labels_train_all)

y_score = knn_model.predict_proba(features_test_all_std)[:, 1]

fpr, tpr, _ = roc_curve(labels_test_all, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, lw=2, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc='lower right')
plt.show()

rf_model = RandomForestClassifier(
    min_samples_split=min_samples_split,
    n_estimators=n_estimators
)
rf_model.fit(features_train_all_std, labels_train_all)

y_score_rf = rf_model.predict_proba(features_test_all_std)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(labels_test_all, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, lw=2, label="Random Forest ROC (AUC = {:.2f})".format(roc_auc_rf))
plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")
plt.show()

svm_model = SVC(C=C_value, kernel='linear', probability=True)
svm_model.fit(features_train_all_std, labels_train_all)

y_score_svm = svm_model.predict_proba(features_test_all_std)[:, 1]

fpr_svm, tpr_svm, _ = roc_curve(labels_test_all, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(8, 8))
plt.plot(fpr_svm, tpr_svm, lw=2, label="SVM ROC (AUC = {:.2f})".format(roc_auc_svm))
plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")
plt.show()

lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(features_train_all_std.shape[1], 1)))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_array = features_train_all_std.to_numpy()
X_test_array = features_test_all_std.to_numpy()
y_train_array = labels_train_all.to_numpy()
y_test_array = labels_test_all.to_numpy()

X_train_lstm = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1], 1)
X_test_lstm = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1], 1)

lstm_model.fit(X_train_lstm, y_train_array, epochs=50,
               validation_data=(X_test_lstm, y_test_array), verbose=0)

y_score_lstm = lstm_model.predict(X_test_lstm).ravel()

fpr_lstm, tpr_lstm, _ = roc_curve(y_test_array, y_score_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

plt.figure(figsize=(8, 8))
plt.plot(fpr_lstm, tpr_lstm, lw=2, label="LSTM ROC (AUC = {:.2f})".format(roc_auc_lstm))
plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")
plt.show()

