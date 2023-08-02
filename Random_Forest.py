import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve


from lightgbm import LGBMClassifier


pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#Random Search


rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

# En iyi hiperparametre değerleri:
rf_random.best_params_

# En iyi skor
rf_random.best_score_


rf_random_final = rf_model.set_params(**rf_random.best_params_,
                                      random_state=17).fit(X, y)


cv_results = cross_validate(rf_random_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

#Grid_Search


rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_,
                               random_state=17).fit(X, y)

cv_results = cross_validate(rf_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# 0.7344326725905673
cv_results['test_f1'].mean()
#  0.5701221536747852
cv_results['test_roc_auc'].mean()
# 0.7710925925925926


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


plot_importance(rf_final, X)


