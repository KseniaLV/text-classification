import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Bidirectional, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def baseline_model(df):
    input_number = df.shape[1] - 1
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(1, input_number)))
    #model.add(GRU(4, input_shape=(1, input_number)))
    model.add(Dense(clusses_number, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


csv_filenames = [
    #"theme_news_char.csv",
    #"theme_news_rhythm.csv",
    #"theme_news_struct.csv",
    #"theme_news_word.csv",
    #"theme_news_bert_avg_by_paragraphs.csv",
    #'theme_news_bert_word.csv',
    #'theme_news_bert_char.csv',
    #'theme_news_bert_rhythm.csv',
    #'theme_news_bert_struct.csv',
    'theme_news_char_struct_word.csv',
    #'theme_news_bert_char_struct_word.csv'
]
result_file = open('multi_classification_LSTM_theme_news_3_themes_errors_3.txt', "w")
for filename in csv_filenames:
    print(filename)
    result_file.write(filename + ' \\\\ \n')
    df = pd.read_csv(filename, header=0, index_col=0)
    for index, row in df.iterrows():
        if index.startswith('Общество') or index.startswith('Наука и технологии') or index.startswith('Политика') or index.startswith('Экономика') or index.startswith('Культура'):# or index.startswith('Медиа'):
            df.drop(index, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    print(df)
    df = (df-df.mean())/df.std()
    df.insert(0, 'ID', range(0, 0 + len(df)))
    labels = [x.split('-')[0].strip() for x in list(df.index.values)]
    categories = list(set(labels))
    clusses_number = len(categories)
    print(categories)
    result_file.write(str(categories) + ' \\\\ \n')
    y = [categories.index(x) for x in labels]
    X = df.to_numpy()
    one_hot_y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_y, random_state = 42, test_size=0.2)
    ids = X_test[:,0] # Сохраняем идентификаторы тестовых данных, чтобы потом анализировать ошибки
    X_train = np.delete(X_train, 0, axis=1)
    X_test = np.delete(X_test, 0, axis=1)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) # для LSTM и GRU
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    model = baseline_model(df)
    model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
    predictions = model.predict(X_test)
    macro = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), average='macro')
    precision = macro[0] * 100
    recall = macro[1] * 100
    f_measure = 2 * precision * recall / (precision + recall)
    result_file.write(str(round(precision, 1))  + ' & ' + str(round(recall, 1))  + ' & ' + str(round(f_measure, 1)) + '\n')
    result_file.write('Errors ' + '\n')
    for text_id, predicted_theme, real_theme in zip(ids, np.argmax(predictions, axis=1), np.argmax(y_test, axis=1)):
        text_name = df.index.values[int(text_id)]
        if real_theme != predicted_theme:
            result_file.write(str(text_name) + ' is ' + str(real_theme) + ', but predicted is ' + str(predicted_theme) + '\n')
    result_file.write('\n')
result_file.close()
