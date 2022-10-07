from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cross_val(classifier):
    results = cross_validate(classifier, X, y, cv=KFold(n_splits=5, shuffle=True), scoring=['precision_macro', 'recall_macro', 'f1_macro'])
    precision = results['test_precision_macro'].mean() * 100
    precision_variability = results['test_precision_macro'].std() * 100
    recall = results['test_recall_macro'].mean() * 100
    recall_variability = results['test_recall_macro'].std() * 100
    f_measure = results['test_f1_macro'].mean() * 100
    f_measure_variability = results['test_f1_macro'].std() * 100
    result_file.write(str(round(precision, 1)) + ' & ' + str(round(precision_variability, 1)) +
                      ' & ' + str(round(recall, 1)) + ' & ' + str(round(recall_variability, 1)) +
                      ' & ' + str(round(f_measure, 1)) + ' & ' + str(round(f_measure_variability, 1)) + ' \\\\ \n')


csv_filenames = [
    "theme_news_char.csv",
    "theme_news_rhythm.csv",
    "theme_news_struct.csv",
    "theme_news_word.csv",
    "theme_news_bert_avg_by_paragraphs.csv",
    'theme_news_bert_word.csv',
    'theme_news_bert_char.csv',
    'theme_news_bert_rhythm.csv',
    'theme_news_bert_struct.csv',
    'theme_news_char_struct_word.csv',
    'theme_news_bert_char_struct_word.csv'
]
result_file = open('multi_class_ml_5_themes_cross_val.txt', "w")

for filename in csv_filenames:
    print(filename)
    df = pd.read_csv(filename, header=0, index_col=0)
    for index, row in df.iterrows():
        if index.startswith('Общество') or index.startswith('Медиа') or index.startswith('Культура'):
            df.drop(index, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    labels = [x.split('-')[0].strip() for x in list(df.index.values)]
    categories = list(set(labels))
    y = [categories.index(x) for x in labels]
    X = df.to_numpy()
    classifiers = [SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1, max_iter=1000), AdaBoostClassifier(n_estimators=50, random_state=100)]
    result_file.write(filename + '\n')
    for classifier in classifiers:
        result_file.write(str(classifier) + '\n')
        cross_val(classifier)
        result_file.write('\n')
result_file.close()
