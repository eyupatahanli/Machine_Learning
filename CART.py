import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile


pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

################################################
# 1. Exploratory Data Analysis
################################################

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

################################################
# 3. Modeling using CART
################################################

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Model
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)


# Confusion matrix için y_pred:
y_pred = cart_model.predict(X)

# AUC için y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=85)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)


# Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# 0.7148
cv_results['test_f1'].mean()
# 0.5780
cv_results['test_roc_auc'].mean()
# 0.6796

# 4. Hyperparameter Optimization with GridSearchCV


cart_model.get_params()

# Arama yapılacak hiperparametre seti:
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)


# En iyi hiperparametre değerleri:
cart_best_grid.best_params_

# En iyi skor:
cart_best_grid.best_score_


random = X.sample(1, random_state=45)

cart_best_grid.predict(random)

# 5. Final Model

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,
                                    random_state=17).fit(X, y)

cart_final.get_params()


# En iyi parametreleri modele atamanın bir diğer yolu:
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)


# Final modelin CV hatası:
cv_results = cross_validate(cart_final,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7344326725905673
cv_results['test_f1'].mean()
#  0.5701221536747852
cv_results['test_roc_auc'].mean()
# 0.7710925925925926

# 6. Feature Importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_final, X, 15)



tree_rules = export_text(cart_model, feature_names=list(X.columns))
print(tree_rules)

################################################
# 10. Extracting Python Codes of Decision Rules
################################################
#todo skompile kütüphanesi gereklilikleri


# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1
# güncel sklearn versiyonu: scikit-learn-0.24.2


# Python kodlarını çıkarmak:
print(skompile(cart_model.predict).to('python/code'))


import sklearn

sklearn.__version__


# SQL kodlarını çıkarmak:
print(skompile(cart_model.predict).to('sqlalchemy/sqlite'))


# Excel kodlarını çıkarmak:
print(skompile(cart_model.predict).to('excel'))

################################################
# 12. Saving and Loading Model
################################################

joblib.dump(cart_final, "cart_final.pkl")
cart_model_from_disk = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]
cart_model_from_disk.predict(pd.DataFrame(x).T)
