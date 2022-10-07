import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('LSTM_bert_confusion_matrix.csv', header=0, index_col=0)
df.loc['Media'] /= 155
df.loc['Sport'] /= 63
df.loc['Science and technologies'] /= 273
df.loc['Culture'] /= 256
df.loc['Politics'] /= 321
df.loc['Economics'] /= 187
df.loc['Society'] /= 295
df.loc['Health'] /= 92
df *= 100
sn.heatmap(df, cbar=True, annot=True, fmt="3.1f", cmap="Greys") #YlGnBu
plt.tight_layout()
plt.savefig('LSTM_bert_confusion_matrix_heatmap.png')
