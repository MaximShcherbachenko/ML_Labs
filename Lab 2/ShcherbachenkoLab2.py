import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'  # your own path to graphviz /bin folder
pd.options.mode.chained_assignment = None  # debug 5th task

# 1. Load the data and store it in dataframe
print('*********************************************************')
print("#1")
path = 'WQ-R.csv'
df = pd.read_csv(path, sep=';')
print("Data has been read")
print('*********************************************************\n')

# 2. Amount of rows and columns
print("#2")
print("Amount of rows and columns:")
print("Rows = {}".format(len(df)))
print("Columns = {}".format(len(df.columns)))
print('*********************************************************\n')

# 3. Print first 10 rows
print("#3")
print("First 10 rows of set:")
print(df.head(10))
print('*********************************************************\n')

# 4. Splitting data into train and test set
print("#4")
x_data = df.drop('quality', axis=1)
y_data = df['quality']
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, random_state=0)
print("Data was split into train and test sets")
print('*********************************************************\n')


# 5. Making instance of decision tree classifier
print("#5")
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)

clf_gini.fit(X_train, Y_train)
clf_entropy.fit(X_train, Y_train)
clf_entropy_pred = clf_entropy.predict(X_test)
clf_gini_pred = clf_gini.predict(X_test)
fn = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
      'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=300)
tree.plot_tree(clf_gini, feature_names=fn, filled=True)
plt.title("Gini tree")
fig.savefig('Tree_gini.png')
plt.show()
fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=300)
tree.plot_tree(clf_entropy, feature_names=fn, filled=True)
plt.title("Entropy tree")
fig.savefig('Tree_entropy.png')
plt.show()
print("Tree images saved (matplotlib)")
print('*********************************************************\n')


# 6. Decision tree representation with Graphviz
print("#6")
dot_data_gini = tree.export_graphviz(clf_gini,
                                     out_file=None,
                                     feature_names=fn,
                                     filled=True)
graph_gini = graphviz.Source(dot_data_gini)
graph_gini.render("graph_gini")  # save image to the current folder

dot_data_entropy = tree.export_graphviz(clf_entropy,
                                        out_file=None,
                                        feature_names=fn,
                                        filled=True)
graph_entropy = graphviz.Source(dot_data_entropy)
graph_entropy.render("graph_entropy")  # save image to the current folder
print("Tree images saved (graphviz)")
print('*********************************************************\n')

# 7.
print("#7")
print("Classification report for training set (criterion='entropy'):\n")
print(classification_report(Y_train, clf_entropy.predict(X_train), zero_division=1))
cm_train_entropy = confusion_matrix(Y_train, clf_entropy.predict(X_train))
print("Confusion matrix for the case above:")
print(cm_train_entropy)
disp_cm_train = ConfusionMatrixDisplay(confusion_matrix=cm_train_entropy, display_labels=clf_entropy.classes_)
disp_cm_train.plot()
plt.title("Confusion matrix for train set (criterion='entropy')")
plt.show()
print('---------------------------------------------------------')

print("\n\nClassification report for training set (criterion='gini'):")
print(classification_report(Y_train, clf_gini.predict(X_train), zero_division=1))
cm_train_gini = confusion_matrix(Y_train, clf_gini.predict(X_train))
print("Confusion matrix for the case above:")
print(cm_train_gini)
disp_cm_train = ConfusionMatrixDisplay(confusion_matrix=cm_train_gini, display_labels=clf_gini.classes_)
disp_cm_train.plot()
plt.title("Confusion matrix for train set (criterion='gini')")
plt.show()
print('---------------------------------------------------------')


print("\n\nClassification report for testing set (criterion='entropy'):\n")
print(classification_report(Y_test, clf_entropy.predict(X_test), zero_division=1))
cm_test_entropy = confusion_matrix(Y_test, clf_entropy.predict(X_test))
print("Confusion matrix for the case above:")
print(cm_test_entropy)
disp_cm_test = ConfusionMatrixDisplay(confusion_matrix=cm_test_entropy, display_labels=clf_entropy.classes_)
disp_cm_test.plot()
plt.title("Confusion matrix for test set (criterion='entropy')")
plt.show()
print('---------------------------------------------------------')


print("\n\nClassification report for testing set (criterion='gini'):")
print(classification_report(Y_test, clf_gini.predict(X_test), zero_division=1))
cm_test_gini = confusion_matrix(Y_test, clf_gini.predict(X_test))
print("Confusion matrix for the case above:")
print(cm_test_gini)
disp_cm_test = ConfusionMatrixDisplay(confusion_matrix=cm_test_gini, display_labels=clf_gini.classes_)
disp_cm_test.plot()
plt.title("Confusion matrix for test set (criterion='gini')")
plt.show()
print('---------------------------------------------------------')

true_pred_dict_gini = {'True': Y_train, 'Pred (gini)': clf_gini_pred}
sns.displot(data=true_pred_dict_gini, color='b', kind='kde')
plt.title("Comparison on the test set (criterion='gini')")
plt.xlabel("quality")
plt.show()
plt.close()
print("Info for 'gini' set:")
unique1, counts1 = np.unique(clf_gini_pred, return_counts=True)
res1 = dict(zip(unique1, counts1))
unique2, counts2 = np.unique(Y_train, return_counts=True)
res2 = dict(zip(unique2, counts2))
print("true {}".format(res2))
print("pred {}".format(res1))
print('---------------------------------------------------------')

true_pred_dict_entropy = {'True': Y_train, 'Pred (entropy)': clf_entropy_pred}
sns.displot(data=true_pred_dict_entropy, color='b', kind='kde')
plt.title("Comparison on the test set (criterion='entropy')")
plt.xlabel("quality")
plt.show()
plt.close()

print("Info for 'entropy' set:")
unique3, counts3 = np.unique(clf_entropy_pred, return_counts=True)
res3 = dict(zip(unique3, counts3))
unique4, counts4 = np.unique(Y_train, return_counts=True)
res4 = dict(zip(unique4, counts4))
print("true {}".format(res4))
print("pred {}".format(res3))
print('---------------------------------------------------------')

print("Accuracy score for entropy criterion = {}. "
      "Amount of correct predictions = {}".format(accuracy_score(Y_test, clf_entropy_pred),
                                                  accuracy_score(Y_test, clf_entropy_pred, normalize=False)))
print("Accuracy score for gini criterion = {}. "
      "Amount of correct predictions = {}".format(accuracy_score(Y_test, clf_gini_pred),
                                                  accuracy_score(Y_test, clf_gini_pred, normalize=False)))
print('*********************************************************\n')


# 8. Tuning the depth of a tree
print("#8")
max_depth_arr = list(range(1, 6))
min_samples_leaf = list(range(1, 6))
accuracy_arr = []
accuracy_arr_acc = []

x_labels = []
for dpth in max_depth_arr:
    for msl in min_samples_leaf:
        x_labels.append("dpth={},msl={}".format(dpth, msl))

for depth in max_depth_arr:
    for leaf in min_samples_leaf:
        clf = DecisionTreeClassifier(criterion='gini', min_samples_leaf=leaf, max_depth=depth, random_state=0)
        clf.fit(X_train, Y_train)

        score = clf.score(X_test, Y_test)
        acc_score = accuracy_score(Y_test, clf.predict(X_test), normalize=False)
        print("Score (depth = {}, min_samples_leaf = {}) = {}. "
              "Amount of correct predictions = {}".format(depth, leaf, score, acc_score))
        accuracy_arr.append(score)

    print("")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_labels, accuracy_arr, label="score")
plt.ylabel("Accuracy")
plt.title("Accuracy score")
fig.autofmt_xdate()
plt.legend()
plt.show()
print('*********************************************************\n')

# 9. Feature importances bar
print("#9")
print("Feature importances list:")
importances = clf_gini.feature_importances_
print(importances)

fig, ax = plt.subplots()
ax.bar(df.columns[:-1], importances, label="feature importances")
fig.autofmt_xdate()
plt.title("Feature importances bar")
plt.ylabel("Mean decrease in impurity")
plt.legend()
plt.show()
