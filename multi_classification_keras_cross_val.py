import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Bidirectional, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def baseline_model():
    input_number = df.shape[1]
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(1, input_number)))
    #model.add(GRU(4, input_shape=(1, input_number)))
    model.add(Dense(clusses_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cross_val(y, df):
    X = df.to_numpy()
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
    results = cross_validate(estimator, X, y, cv=KFold(n_splits=3, shuffle=True), scoring=['precision_macro', 'recall_macro', 'f1_macro'])
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
    #"theme_news_char.csv",
    #"theme_news_rhythm.csv",
    #"theme_news_struct.csv",
    #"theme_news_word.csv",
    "theme_news_bert_avg_by_paragraphs.csv",
    #'theme_news_bert_word.csv',
    #'theme_news_bert_char.csv',
    #'theme_news_bert_rhythm.csv',
    #'theme_news_bert_struct.csv',
    'theme_news_char_struct_word.csv',
    'theme_news_bert_char_struct_word.csv'
]
result_file = open('multi_classification_LSTM_theme_news_8_themes_4_feat_types_cross_val.txt', "w")
for filename in csv_filenames:
    print(filename)
    result_file.write(filename + ' \\\\ \n')
    df = pd.read_csv(filename, header=0, index_col=0)
    for index, row in df.iterrows():
        if index.startswith('Общество') or index.startswith('Медиа') or index.startswith('Культура'):
            df.drop(index, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    labels = [x.split('-')[0].strip() for x in list(df.index.values)]
    categories = list(set(labels))
    clusses_number = len(categories)
    print(categories)
    result_file.write(str(categories) + ' \\\\ \n')
    y = [categories.index(x) for x in labels]
    df = (df-df.mean())/df.std()
    cross_val(y, df)
    result_file.write('\n')
result_file.close()
