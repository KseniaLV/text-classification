from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


lang = "ru"
csv_filenames = [
    "ru_rhythm_stat.csv",
    "ru_char.csv",
    "ru_word.csv"
]
result_file = open('AdaBoost_verification_with_pca_'+ lang+'.txt', "w")


def getFeatureTypeByFilename(filename):
    if "_all_features" in filename:
        return "7-All"
    elif "_word_and_rhythm" in filename:
        return "6-W + Rh"
    elif "_char_and_rhythm" in filename:
        return "5-Ch + Rh"
    elif "_char_and_word" in filename:
        return "4-Ch + W"
    elif "_rhythm" in filename:
        return "3-Rh"
    elif "_word" in filename:
        return "2-W"
    else:
        return "1-Ch"


def cross_val(filename):
    #classifier = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=100)
    classifier = AdaBoostClassifier(n_estimators=50, random_state=100)
    results = cross_validate(classifier, X, y, cv=KFold(n_splits=5, shuffle=True), scoring=['precision_macro', 'recall_macro', 'f1_macro'])
    precision = results['test_precision_macro'].mean() * 100
    precision_variability = results['test_precision_macro'].std() * 100
    recall = results['test_recall_macro'].mean() * 100
    recall_variability = results['test_recall_macro'].std() * 100
    f_measure = results['test_f1_macro'].mean() * 100
    f_measure_variability = results['test_f1_macro'].std() * 100
    precisions.append(precision)
    recalls.append(recall)
    result_file.write(author + ' & ' + getFeatureTypeByFilename(filename) + ' & ' + str(round(precision, 1)) + ' & ' + str(round(precision_variability, 1)) +
                      ' & ' + str(round(recall, 1)) + ' & ' + str(round(recall_variability, 1)) +
                      ' & ' + str(round(f_measure, 1)) + ' & ' + str(round(f_measure_variability, 1)) + ' \\\\ \n')


for filename in csv_filenames:
    df = pd.read_csv(filename, header=0, index_col=0)
    result_file.write(filename + '\n')
    
    classes_labels = [index.split('-')[1].strip() for index, row in df.iterrows()]
    unique_classes = set(classes_labels)
    result_file.write(str(unique_classes) + '\n')
    df = (df - df.mean()) / df.std()
    X = df.to_numpy()

    precisions = []
    recalls = []
    for author in unique_classes:
        print(author)
        result_file.write('Verify ' + author + '\n')
        y = [0 if label == author else 1 for label in classes_labels]
        result_file.write('Number of fragments: ' + str(len(y) - sum(y)) + '\n')
        cross_val(filename)
    result_file.write('Average metrics\n' + getFeatureTypeByFilename(filename) + ' & ')
    mean_precision = sum(precisions) / len(precisions)
    result_file.write(str(round(mean_precision, 1)) + ' & ')
    mean_recall = sum(recalls) / len(recalls)
    result_file.write(str(round(mean_recall, 1)) + ' & ')
    mean_f_measure = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
    result_file.write(str(round(mean_f_measure, 1)) + '\n')
    result_file.write('\n')
result_file.close()
