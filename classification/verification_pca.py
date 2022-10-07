from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

langs = ["fr", "es", "en", "ru"]

for lang in langs:
    csv_filenames = [
    lang + "_char.csv",
    lang + "_word.csv", lang + "_rhythm.csv"
                 ]
    for filename in csv_filenames:
        df = pd.read_csv(filename, header=0, index_col=0)    
        classes_labels = [index.split('-')[1].strip() for index, row in df.iterrows()]
        unique_classes = set(classes_labels)
        x = StandardScaler().fit_transform(df)
        n_components = len(df.columns)
        #if filename == lang + "_rhythm.csv":
            #n_components = len(df.columns) - 5
        #elif filename == lang + "_all_features.csv":
            #n_components=len(df.columns) - 35
        #else:
            #n_components=len(df.columns) - 15
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(x)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        PC_values = np.arange(pca.n_components_) + 1
        plt.figure(figsize=(6, 6))
        plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=1)
        plt.plot(PC_values, cumulative, 'go-', linewidth=1)
        #plt.title('Scree Plot: ' + filename)
        plt.xlabel('Компоненты')
        plt.ylabel('Доля информации (доля от общей объяснённой дисперсии)')
        plt.savefig(filename[:-4] + '_scree_plot.png')
        plt.clf()
    #principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2'])
    #for author in unique_classes:
        #print(author)
        #y = [label if label == author else "no" for label in classes_labels]
        #finalDf = pd.concat([principalDf, pd.DataFrame(data=y, columns=['target'])], axis = 1)
        #fig = plt.figure(figsize = (8,8))
        #ax = fig.add_subplot(1,1,1) 
        #ax.set_xlabel('Principal Component 1', fontsize = 15)
        #ax.set_ylabel('Principal Component 2', fontsize = 15)
        #ax.set_title('2 component PCA', fontsize = 20)
        #targets = [author, 'no']
        #colors = ['r', 'g']
        #for target, color in zip(targets,colors):
            #indicesToKeep = finalDf['target'] == target
            #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'] , finalDf.loc[indicesToKeep, 'principal component 2'] , c = color , s = 50)
        #ax.legend(targets)
        #plt.savefig(filename[:-4] + '_' + author + '.png')
        #plt.clf()

        
        
