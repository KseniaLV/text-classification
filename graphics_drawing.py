import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('diff_themes_features.csv', header=0)
    for quality in ['Точность','Полнота','F-мера']:
        plt.figure(figsize=(7, 7))
        plt.locator_params(axis='x', nbins=7)
        plt.locator_params(axis='y', nbins=40)
        line_types = ['-k', '--k', '-.k', ':k']
        for i, feature in enumerate(['BERT', 'Char+Struct+Word', 'BERT+Char+Struct+Word']):
            filtered = df.loc[df['Характеристика'] == feature]
            theme_number = filtered['Количество тем'].tolist()
            feat_values = filtered[quality].tolist()
            plt.plot(theme_number, feat_values, line_types[i], label=feature)
        plt.tight_layout()
        plt.legend()
        plt.savefig(quality + '.png')
        plt.clf()
