import pandas as pd

#read in word_library as word_src
#w = pd.read_table('sgns.merge.word', sep=' ', skiprows=[0], nrows=200, header=None, index_col=0)
"""
word_src = pd.read_table('sgns.merge.word', sep=' ', skiprows=[0], header=None, index_col=0)
word_src.drop(301, axis=1, inplace=True)
"""
#data = pd.read_csv()
#data['内容']
data = pd.read_csv('2、训练集_选取所有的维稳数据作为正样本.csv')
#print(data['归口'])
for i in range(0, 23628):
    #分词
