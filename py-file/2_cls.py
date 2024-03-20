# Python Script for Machine Learning Tasks
import os
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, linear_model
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Digraph
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

# Need to change path
os.chdir(r"C:\Users\avis1\Desktop\master_thesis")


# import training dataset
data_ml = r'.\wdata\mu_ML.dta'
df = pd.read_stata(data_ml)
X = df[['gender', 'age', "kid03", "kid35", "kid615", "kid1517", "kid18",
        "edu1", "edu2", "edu3", "edu4", "edu5", "edu6", "edu7", "edu8", "edu9",
        "rel1", "rel2", "rel3", "rel4", "rel5", "rel6", "rel7", "rel8", "rel9", "rel10", "rel11", "rel12", "rel13", "rel14",
        "mar1", "mar2", "mar3", "mar4",
        "county1", "county2", "county3", "county4", "county5", "county6", "county7", "county8", "county9",
        "county10", "county11", "county12", "county13", "county14", "county15", "county16", "county17",
        "county18", "county19", "county20",
        "major1", "major2", "major3", "major4", "major5", "major6", "major7", "major8", "major9", "major10", "major11"]]
y = df['treat']


# import prediction dataset
data_pd = r'.\wdata\mu_PD.dta'
df_pd = pd.read_stata(data_pd)
X_pd = df_pd[['gender',  'age',  "kid03", "kid35", "kid615", "kid1517", "kid18",
              "edu1", "edu2", "edu3", "edu4", "edu5", "edu6", "edu7", "edu8", "edu9",
              "rel1", "rel2", "rel3", "rel4", "rel5", "rel6", "rel7", "rel8", "rel9", "rel10", "rel11", "rel12", "rel13", "rel14",
              "mar1", "mar2", "mar3", "mar4",
              "county1", "county2", "county3", "county4", "county5", "county6", "county7", "county8", "county9",
              "county10", "county11", "county12", "county13", "county14", "county15", "county16", "county17",
              "county18", "county19", "county20",
              "major1", "major2", "major3", "major4", "major5", "major6", "major7", "major8", "major9", "major10", "major11"]]


# split X into training and testing sets, 7:3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

df_train = df.iloc[y_train.index]
df_valid = df.iloc[y_test.index]

df_train.to_stata(r'.\wdata\mu_train.dta', version=118)
df_valid.to_stata(r'.\wdata\mu_valid.dta', version=118)

# ------------------------------------------------------------------------------------------ logi

logi = LogisticRegression(penalty='l2', 
                          dual=False,
                          tol=0.0001, 
                          C=1.0, 
                          fit_intercept=True, 
                          intercept_scaling=1, 
                          class_weight=None, 
                          random_state=None, 
                          solver='newton-cg', 
                          max_iter=100, 
                          multi_class='auto', 
                          verbose=0, 
                          warm_start=False, 
                          n_jobs=None, 
                          l1_ratio=None)
logi.fit(X_train, y_train)

# ------------------------------------------------------------------------------------------ Tree
tree = DecisionTreeClassifier( criterion='entropy',
                               splitter='best',
                               max_depth=5,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features='sqrt',
                               random_state=0,
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               class_weight=None,
                               ccp_alpha=0.0)

tree.fit(X_train, y_train)

if not os.path.exists("./output/") : os.mkdir("./output/")
export_graphviz(
    tree,
    out_file='./output/tree.dot',
    feature_names=X.columns.values
)
# dot -Tpng C:\Users\avis1\Desktop\master_thesis\output\tree.dot -o C:\Users\avis1\Desktop\master_thesis\output\fig-tree.png

#------------------------------------------------------------------------------------------- Bagging
bagging = BaggingClassifier(n_estimators=100,
                            random_state=0,
                            max_samples=65705,
                            max_features=20)
bagging.fit(X_train, y_train)

#------------------------------------------------------------------------------------------- Random Forests
forest = RandomForestClassifier(n_estimators=100,
                                criterion='entropy',
                                max_depth=5,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features='sqrt',
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=None,
                                random_state=0,
                                verbose=0,
                                warm_start=False,
                                class_weight=None,
                                ccp_alpha=0.0,
                                max_samples=None)
forest.fit(X_train, y_train)

#------------------------------------------------------------------------------------------- AdaBoost
# AdaBoost Model
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1.0,
                         random_state=0
                         )
# Train Adaboost Classifer
abc.fit(X_train, y_train)

#------------------------------------------------------------------------------------------- Gradient Boosting
GBC = GradientBoostingClassifier(learning_rate=0.1,
                                 n_estimators=100,
                                 subsample=1.0,
                                 criterion='friedman_mse',
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_depth=3,
                                 min_impurity_decrease=0.0,
                                 init=None,
                                 random_state=0,
                                 max_features=None,
                                 verbose=0,
                                 max_leaf_nodes=None,
                                 warm_start=False,
                                 validation_fraction=0.1,
                                 n_iter_no_change=None,
                                 tol=0.0001,
                                 ccp_alpha=0.0)
GBC.fit(X_train, y_train)

#------------------------------------------------------------------------------------------- MLP
MLP = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=500,
                    shuffle=True,
                    random_state=0,
                    tol=0.0001,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08,
                    n_iter_no_change=10,
                    max_fun=15000
                    )
MLP.fit(X_train, y_train)


# six model name
model_name = ['Logistic Regression', 'Decision Trees', 'Bagging Decision Trees', 'Random Forest', 'AdaBoost Decision Trees', 'Gradient Boosting Decision Trees', 'Multi-layer Perceptron']

# print accuracy
def print_accuracy(model, y_pred_train, y_pred_test):
    print('Accuracy ('+ model+ ', train): %.4f' % accuracy_score(y_train, y_pred_train))
    print('Accuracy ('+ model+ ', test): %.4f' % accuracy_score(y_test, y_pred_test))

print_accuracy(model_name[0], logi.predict(X_train), logi.predict(X_test))
print_accuracy(model_name[1], tree.predict(X_train), tree.predict(X_test))
print_accuracy(model_name[2], bagging.predict(X_train), bagging.predict(X_test))
print_accuracy(model_name[3], forest.predict(X_train), forest.predict(X_test))
print_accuracy(model_name[4], abc.predict(X_train), abc.predict(X_test))
print_accuracy(model_name[5], GBC.predict(X_train), GBC.predict(X_test))
print_accuracy(model_name[6], MLP.predict(X_train), MLP.predict(X_test))



# Precision-Recall Curve (Treatment group)
Y_test = label_binarize(y_test, classes=[*range(3)])
pr_treated = {}
prob_treated = [logi.predict_proba(X_test)[:, 0],
                tree.predict_proba(X_test)[:, 0], 
                bagging.predict_proba(X_test)[:, 0],
                forest.predict_proba(X_test)[:, 0],
                abc.predict_proba(X_test)[:, 0],
                GBC.predict_proba(X_test)[:, 0],
                MLP.predict_proba(X_test)[:, 0]]
plt.figure(figsize=(8,6))
for i in range(7):
    precision, recall, thresholds = precision_recall_curve(Y_test[:, 0], prob_treated[i])
    plt.scatter(recall, precision, label=model_name[i], s=0.05)
    thresholds = thresholds.tolist()
    thresholds.append(1)
    pr_treated[model_name[i]] = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc="upper right", markerscale=15)
plt.savefig(r'.\pic\PR_curve_t.png', dpi=300)
plt.show()

# Precision-Recall Curve (Control group)
pr_control = {}
prob_control = [logi.predict_proba(X_test)[:, 1],
                tree.predict_proba(X_test)[:, 1], 
                bagging.predict_proba(X_test)[:, 1],
                forest.predict_proba(X_test)[:, 1],
                abc.predict_proba(X_test)[:, 1],
                GBC.predict_proba(X_test)[:, 1],
                MLP.predict_proba(X_test)[:, 1]]
plt.figure(figsize=(8,6))
for i in range(7):
    precision, recall, thresholds = precision_recall_curve(Y_test[:, 1], prob_control[i])
    plt.scatter(recall, precision, label=model_name[i], s=0.05)
    thresholds = thresholds.tolist()
    thresholds.append(1)
    pr_control[model_name[i]] = pd.DataFrame({'precision': precision, 'recall': recall, 'thresholds': thresholds})

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc="upper right", markerscale=15)
plt.savefig(r'.\pic\PR_curve_c.png', dpi=300)
plt.show()

def find_index_of_first_larger_than_k(number_list, k):
    for index, number in enumerate(number_list):
        if number > k:
            return index  # Found the index of the first number larger than k

    return -1  # No number larger than k found in the list




pr_control[model_name[5]] = pr_control[model_name[5]].sort_values(by='recall')
pr_treated[model_name[5]] = pr_treated[model_name[5]].sort_values(by='recall')

pr_control[model_name[5]] = pr_control[model_name[5]].reset_index(drop=True)
pr_treated[model_name[5]] = pr_treated[model_name[5]].reset_index(drop=True)


gbc_thresholds_c = pr_control[model_name[5]]['thresholds'][find_index_of_first_larger_than_k(pr_control[model_name[5]]['recall'], 0.75)]
gbc_thresholds_t = pr_treated[model_name[5]]['thresholds'][find_index_of_first_larger_than_k(pr_treated[model_name[5]]['recall'], 0.75)]


prob_treated_pd = [ logi.predict_proba(X_pd)[:, 0],
                    tree.predict_proba(X_pd)[:, 0], 
                    bagging.predict_proba(X_pd)[:, 0],
                    forest.predict_proba(X_pd)[:, 0],
                    abc.predict_proba(X_pd)[:, 0],
                    GBC.predict_proba(X_pd)[:, 0],
                    MLP.predict_proba(X_pd)[:, 0]]

prob_control_pd = [ logi.predict_proba(X_pd)[:, 1],
                    tree.predict_proba(X_pd)[:, 1], 
                    bagging.predict_proba(X_pd)[:, 1],
                    forest.predict_proba(X_pd)[:, 1],
                    abc.predict_proba(X_pd)[:, 1],
                    GBC.predict_proba(X_pd)[:, 1],
                    MLP.predict_proba(X_pd)[:, 1]]

prob_other_pd = [ logi.predict_proba(X_pd)[:, 2],
                    tree.predict_proba(X_pd)[:, 2], 
                    bagging.predict_proba(X_pd)[:, 2],
                    forest.predict_proba(X_pd)[:, 2],
                    abc.predict_proba(X_pd)[:, 2],
                    GBC.predict_proba(X_pd)[:, 2],
                    MLP.predict_proba(X_pd)[:, 2]]


df_pd['treat_GBC_t'] = 0     
df_pd['treat_GBC_c'] = 0  
df_pd.loc[prob_treated_pd[5] >= gbc_thresholds_t, 'treat_GBC_t'] = 1
df_pd.loc[prob_control_pd[5] >= gbc_thresholds_c, 'treat_GBC_c'] = 1



df_pd.to_stata(r'.\wdata\mu_MLPD.dta', version=118)

