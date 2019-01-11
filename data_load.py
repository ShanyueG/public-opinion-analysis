import pandas as pd
import pkuseg
import numpy as np
import torch


def split_sentence(sentence):
    text = seg.cut(sentence)
    return text


# read in word_library as word_src
word_src = pd.read_table('sgns.merge.word', sep=' ', nrows=300, skiprows=[0], header=None, index_col=0)
word_src.drop(301, axis=1, inplace=True)


data = pd.read_csv('2、训练集_选取所有的维稳数据作为正样本.csv')
# print(data.loc[1, '归口'])
# data.loc['行名', '列名']
# data.iLoc[1:2,3:2]


sentence = []
word_vec = []
label = [0]*23629
max_size = 0
all_sent = [0]*23629

seg = pkuseg.pkuseg()

for i in range(0, 10):
    # label
    word_vec = []
    if '社会维稳' in data.loc[i, '归口']:
        label[i] = 1
    # split sentence
    sentence = data.loc[i, '内容']
    all_words = split_sentence(sentence)
    for j in range(0, len(all_words)):
        word = all_words[j]
        try:
            word_vec.append(word_src.loc[word].tolist())
        except:
            word_vec.append([0]*300)

    sent_array = np.array(word_vec)
    all_sent[i] = sent_array

# calculate the max_len of all sentences, update every routes
    if len(all_words) > max_size:
        max_size = len(all_words)

# reshape
for i in range(0, 10):
    num = np.size(all_sent[i], 0)
    if num < max_size:
        append_ary = np.zeros((max_size-num, 300))
        all_sent[i] = np.vstack((all_sent[i], append_ary))

all_sent = np.array(all_sent[:10])

all_sent = torch.Tensor(all_sent)
print('write over')