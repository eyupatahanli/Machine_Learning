import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df = pd.read_csv("datasets/diabetes.csv")
# Tüm sayısal değişkenlerin özet istatistikleri:
df.describe().T

# Hedef değişkenin sınıfları ve frekansları:
df["Outcome"].value_counts()

# Frekanslar görsel olarak
sns.countplot(x="Outcome", data=df)
plt.show()

# Hedef değişkenin sınıf oranları:
100 * df["Outcome"].value_counts() / len(df)

#değişkenlerin analizi

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

# bütün değişkenler sayısal olduğu için hepsine uygulanabilir
for col in df.columns:
    plot_numerical_col(df, col)


cols = [col for col in df.columns if "Outcome" not in col]

#hedef değişkene göre sayısal değişkenlerin ortalaması
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)

# Data Preprocessing (Veri Ön İşleme)

# Eksik değer incelemesi:
df.isnull().sum()
#todo eksik değer çalışmaları

# Aykırı değer incelemesi:

#Todo: aykırı değer çalışmaları

# Model & Prediction

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Model:
log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

# Tahmin
y_pred = log_model.predict(X)

# Model Evaluation
print(classification_report(y, y_pred))
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# Model Validation: Holdout
# Veri setinin train-test olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)


# Modelin train setine kurulması:
log_model = LogisticRegression().fit(X_train, y_train)

# Test setinin modele sorulması:
y_pred = log_model.predict(X_test)

# AUC Score için y_prob (1. sınıfa ait olma olasılıkları)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Classification report
print(classification_report(y_test, y_pred))

roc_auc_score(y_test, y_prob)

# Model Validation: 10-Fold Cross Validation

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

