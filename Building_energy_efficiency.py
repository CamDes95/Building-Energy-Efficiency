#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("ENB.csv", sep = "\t", header=0, decimal=",")
print(df.info())
print(df.head())


# Analyse des corrélations entre les variables de df

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, ax = ax, cmap="Spectral")

# Variables explicatives les plus correlées avec variables cibles :
    # heating_load : corr positive avec cooling_head et overall_height
    #                corr négative avec roof_area

    # cooling_load : corr positive avec heating_load et overall_height
    #                corr négative avec roof_area


# In[39]:


# Nouvelle colonne de df total_charges
df["total_charges"] = df["cooling_load"] + df["heating_load"]
print(df.describe(), "\n")

# définition des quantiles
q1 = df["total_charges"].quantile(q = 0.25)
q2 = df["total_charges"].quantile(q = 0.5)
q3 = df["total_charges"].quantile(q = 0.75)

charges_classes = pd.cut(df["total_charges"], [16, q1, q2, q3, 90], labels = [0,1,2,3])

print(charges_classes.value_counts().sort_values(ascending=True))


# remplacement des Nan par mode
print(charges_classes.isna().sum())


# Séparation des variables explicatives
data = df.drop(["heating_load", "cooling_load"], axis=1)

# Séparation des données en train set et test set
X_train, X_test, y_train, y_test = train_test_split(data, charges_classes, test_size = 0.2, random_state = 123)

# Standardisation des variables explicatives
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

scaler2 = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler2.transform(X_test)




## KNN : n_neighbors de 2 à 50
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors = 5)

clf_knn.fit(X_train_scaled, y_train)
y_pred = clf_knn.predict(X_test_scaled)
pd.crosstab(y_test, y_pred)
clf_knn.score(X_test_scaled, y_test) # taux bonnes prédictions


# grille de recherche 
from sklearn import model_selection

param_knn = {"n_neighbors": range(2,50)}
grid_clf_knn = model_selection.GridSearchCV(estimator = clf_knn, param_grid = param_knn)
grille_knn= grid_clf_knn.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grille_knn.cv_results_).loc[:,["params","mean_test_score"]], "\n")

print("Meilleurs paramètres de knn : {}".format(grid_clf_knn.best_params_))





## SVM : kernel : "rbf", "linear", c = [0.1, 1, 10, 50]
from sklearn import svm

clf_svm = svm.SVC(gamma = 0.01, kernel = "linear", C=10)

clf_svm.fit(X_train_scaled, y_train)
y_pred = clf_svm.predict(X_test_scaled)
pd.crosstab(y_test, y_pred)
clf_svm.score(X_test_scaled, y_test)

# grille de recherche 
param_svm = {"C": [0.1, 1, 10, 50], "kernel":["rbf", "linear"]}
grid_clf_svm = model_selection.GridSearchCV(estimator = clf_svm, param_grid = param_svm)
grille_svm = grid_clf_svm.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grille_svm.cv_results_).loc[:,["params","mean_test_score"]],"\n")

print("Meilleur hyperparamètres de svm : {} ".format(grid_clf_svm.best_params_))




## RandomForest : max_features : "sqrt", "log2", None
#                min_samples_split : nombres pairs de 2 à 30
from sklearn.ensemble import RandomForestClassifier

clf_ra = RandomForestClassifier(n_jobs = -1)

clf_ra.fit(X_train_scaled, y_train)
y_pred = clf_ra.predict(X_test_scaled)
pd.crosstab(y_test, y_pred)
clf_ra.score(X_test_scaled, y_test) # taux bonnes prédictions

# grille de recherche 
param_ra = {"max_features": ["sqrt", "log2", None], "min_samples_split":range(2,30,2)}
grid_clf_ra = model_selection.GridSearchCV(estimator = clf_ra, param_grid = param_ra)
grille_ra = grid_clf_ra.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grille_ra.cv_results_).loc[:,["params","mean_test_score"]])

print("Meilleurs hyperparamètres pour Random Forest : {} ".format(grid_clf_ra.best_params_))




# Création d'une méthode d'ensemble VotingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
import sklearn

vc = VotingClassifier(estimators = [("knn", clf_knn), ("svm", clf_svm),("ra", clf_ra)], voting= "hard")
cv3 = KFold(n_splits = 3, random_state = 122, shuffle = True) # cross-validator 3 parties

clfs = [clf_knn, clf_svm, clf_ra, vc]
labels = ["KNN", "SVM", "RandomForest", "VotingClassifier"]
scoring = {'accuracy': make_scorer(sklearn.metrics.accuracy_score),
           'f1_macro': make_scorer(sklearn.metrics.f1_score, average = 'weighted')}

for clf, label in zip(clfs, labels):
    score = cross_validate(clf, X_train_scaled, y_train, cv = cv3, scoring = scoring)
    print("{} : \n Accuracy : {} (+/- {}) \n F1_Score : {} (+/- {})".format(
        label,
        np.round(score["test_accuracy"].mean(), 3),
        np.round(score["test_accuracy"].std(), 3),
        np.round(score["test_f1_macro"].mean(), 3),
        np.round(score["test_f1_macro"].std(), 3)))



# perfs du votingclassifier
vc.fit(X_train_scaled, y_train)
print("VotingClassifier score : ",vc.score(X_test_scaled, y_test))

# comparaison avec meilleur modèle randomforest
clf_ra.fit(X_train_scaled, y_train)
print("RandomForest score :",clf_ra.score(X_test_scaled, y_test))

# Le modèle d'ensemble permet de gagner 0.5 pts d'accuracy /r à l'utilisation d'un seul modèle





