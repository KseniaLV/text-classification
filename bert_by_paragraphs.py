import os
import pandas as pd
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import re
import numpy
import shutil


def clear_line(line):
    trimmed_line = line.replace('*', '').strip()
    if not trimmed_line:
        return trimmed_line
    line_without_multi_spaces = ' '.join(trimmed_line.split())
    clean_line = line_without_multi_spaces.replace('"', '\'')
    clean_line = clean_line.replace(' ', ' ')
    clean_line = clean_line.replace('—', '-')
    clean_line = clean_line.replace('–', '-')
    clean_line = clean_line.replace('−', '-')
    clean_line = re.sub('-+', '-', clean_line)
    clean_line = clean_line.replace(' :', ':')
    clean_line = clean_line.replace('« ', '\'')
    clean_line = clean_line.replace(' »', '\'')
    clean_line = clean_line.replace('“', '\'')
    clean_line = clean_line.replace('”', '\'')
    clean_line = clean_line.replace(' ?', '?')
    clean_line = clean_line.replace(' ;', ';')
    clean_line = clean_line.replace(' !', '!')
    clean_line = clean_line.replace(' …', '…')
    clean_line = clean_line.replace('’', "'")
    clean_line = clean_line.replace('‘', "'")
    clean_line = clean_line.replace("-'", "'")
    clean_line = clean_line.replace("_", "")
    clean_line = re.sub(r'([.?!…]+)([\"\'])([ \n]?)', r'\2\1\3', clean_line)
    return clean_line


os.environ["KERAS_BACKEND"] = "tensorflow"
bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['variables']['BERT_PATH'] = '/home/ksenia/.deeppavlov/downloads/embeddings/rubert_cased_L-12_H-768_A-12_pt'

m = build_model(bert_config)
print('Bert model!')

names = []
TEXTS_DIR = 'ru_smnovels'
all_embeddings = []
for filename in os.listdir(TEXTS_DIR):
    full_path = os.path.join(TEXTS_DIR, filename)
    if os.path.isfile(full_path) and filename.endswith('txt'):
        print("Processing " + filename)
        with open(full_path) as fp:
            paragraphs = []
            line = fp.readline()
            while line:
                clean_line = clear_line(line)
                if clean_line:
                    tokens = clean_line.split()
                    #if (len(tokens) > 500):
                        #clean_line = " ".join(tokens[:501])
                        #print(clean_line)
                    paragraphs.append(clean_line)
                line = fp.readline()
        try:
            tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = m(paragraphs)
            a = numpy.array(sent_mean_embs)
            all_embeddings.append(numpy.mean(a, axis=0))
            names.append(filename)
        except Exception as e:
            print('Exception: '+ str(e))
            print(filename)
            shutil.copyfile(full_path, os.path.join('recount', filename))

df = pd.DataFrame(all_embeddings, index=names)
df.to_csv(TEXTS_DIR + '_bert_avg_by_paragraphs.csv', index=True)
