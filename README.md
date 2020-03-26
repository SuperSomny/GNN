x是训练集中已标注节点的特征矩阵
tx是测试集的特征矩阵
allx是训练集中所有节点的特征矩阵

上述特征矩阵的类型是scipy.sparse.csr.csr_matrix

y是训练集中已标注节点的标签(one-hot编码)
ty是测试集的标签
ally是训练集中所有节点的标签

上述标签的类型是numpy.ndarray

graph是整个图的邻接矩阵(训练集+测试集)，类型是dict
test.index是测试集中节点的索引

上述所有数据都要用pickle存储在data目录下对应的文件中
