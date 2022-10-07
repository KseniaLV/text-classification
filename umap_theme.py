import pandas as pd
import umap
import umap.plot
import matplotlib.pyplot as plt

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
    #'theme_news_char_struct_word.csv',
    #'theme_news_bert_char_struct_word.csv'
]

for filename in csv_filenames:
    df = pd.read_csv(filename, header=0, index_col=0)
    #for index, row in df.iterrows():
        #if index.startswith('Общество') or index.startswith('Медиа') or index.startswith('Культура'):
            #df.drop(index, inplace=True)
    df = df.loc[:, (df != 0).any(axis=0)]
    category_labels = [x.split('-')[0].strip() for x in list(df.index.values)]
    hover_df = pd.DataFrame(category_labels, columns=['category'])
    features_matrix = df.to_numpy()
    for metric in ['euclidean', 'chebyshev', 'correlation']:
        embedding = umap.UMAP(n_components=2, n_neighbors=20, metric=metric, min_dist=0.1).fit(features_matrix)
        umap.plot.connectivity(embedding, show_points=True)
        #umap.plot.points(embedding, labels=hover_df['category'], theme='viridis')
        save_name = 'connectivity_' + filename[:-4] + '_' + metric + '_umap.png'
        plt.savefig(save_name)
        print(save_name)
        plt.clf()
