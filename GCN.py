#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import warnings
import scipy.sparse as sp
from time import time
from sklearn.metrics import accuracy_score
import tensorflow as tf
from collections import defaultdict
import pickle
import networkx as nx


# In[2]:


from easydict import EasyDict

config = {
    'dataset': 'cora',
    'hidden1': 16,
    'epochs': 200,
    'early_stopping': 20,
    'weight_decay': 5e-4,
    'learning_rate': 0.01,
    'dropout': 0.,
    'verbose': False,
    'logging': False,
    'gpu_id': None
}

FLAGS = EasyDict(config)


# # 辅助函数
# 
# ## 数据读取

# In[3]:


def load_data_planetoid(dataset):
    keys = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = defaultdict()
    for key in keys:
        with open('data_split/ind.{}.{}'.format(dataset, key), 'rb') as f:
            objects[key] = pickle.load(f, encoding='latin1')
    test_index = [int(x) for x in open('data_split/ind.{}.test.index'.format(dataset))]
    test_index_sort = np.sort(test_index)
    G = nx.from_dict_of_lists(objects['graph'])

    A_mat = nx.adjacency_matrix(G)
    X_mat = sp.vstack((objects['allx'], objects['tx'])).tolil()
    X_mat[test_index, :] = X_mat[test_index_sort, :]
    z_vec = np.vstack((objects['ally'], objects['ty']))
    z_vec[test_index, :] = z_vec[test_index_sort, :]
    z_vec = z_vec.argmax(1)

    train_idx = range(len(objects['y']))
    val_idx = range(len(objects['y']), len(objects['y']) + 500)
    test_idx = test_index_sort.tolist()

    return A_mat, X_mat, z_vec, train_idx, val_idx, test_idx


# ## 用于处理稀疏矩阵

# In[5]:


# 稀疏矩阵的 dropout
def sparse_dropout(x, dropout_rate, noise_shape):
    random_tensor = 1 - dropout_rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    # 从稀疏矩阵中取出dropout_mask对应的元素
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / (1 - dropout_rate))

# 稀疏矩阵转稀疏张量
def sp_matrix_to_sp_tensor(M):
    if not isinstance(M, sp.csr.csr_matrix):
        M = M.tocsr()
    # 获取非0元素坐标
    row, col = M.nonzero()
    # SparseTensor参数：二维坐标数组，数据，形状
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X


# # 定义图卷积层

# In[6]:


import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers

class GCNConv(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(GCNConv, self).__init__()

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)


    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        fdim = input_shape[1][1]  # feature dim
        # 初始化权重矩阵
        self.weight = self.add_weight(name="weight",
                                      shape=(fdim, self.units),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            # 初始化偏置项
            self.bias = self.add_weight(name="bias",
                                        shape=(self.units, ),
                                        initializer=self.bias_initializer,
                                        trainable=True)

    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]
        # 计算 XW
        if isinstance(self.X, tf.SparseTensor):
            h = tf.sparse.sparse_dense_matmul(self.X, self.weight)
        else:
            h = tf.matmul(self.X, self.weight)
        # 计算 AXW
        output = tf.sparse.sparse_dense_matmul(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output


# # 定义GCN模型

# In[7]:


tf.get_logger().setLevel('ERROR')

class GCN():
    def __init__(self, An, X, sizes, **kwargs):
        self.with_relu = True
        self.with_bias = True

        self.lr = FLAGS.learning_rate
        self.dropout = FLAGS.dropout
        self.verbose = FLAGS.verbose
        
        self.An = An
        self.X = X
        self.layer_sizes = sizes
        self.shape = An.shape

        self.An_tf = sp_matrix_to_sp_tensor(self.An)
        self.X_tf = sp_matrix_to_sp_tensor(self.X)

        self.layer1 = GCNConv(self.layer_sizes[0], activation='relu')
        self.layer2 = GCNConv(self.layer_sizes[1])
        self.opt = tf.optimizers.Adam(learning_rate=self.lr)

    def train(self, idx_train, labels_train, idx_val, labels_val):
        K = labels_train.max() + 1
        train_losses = []
        val_losses = []
        # use adam to optimize
        for it in range(FLAGS.epochs):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(idx_train, np.eye(K)[labels_train])

            # optimize over weights
            grad_list = tape.gradient(_loss, self.var_list)
            grads_and_vars = zip(grad_list, self.var_list)
            self.opt.apply_gradients(grads_and_vars)

            # evaluate on the training
            train_loss, train_acc = self.evaluate(idx_train, labels_train, training=True)
            train_losses.append(train_loss)
            val_loss, val_acc = self.evaluate(idx_val, labels_val, training=False)
            val_losses.append(val_loss)
            toc = time()
            if self.verbose:
                print("iter:{:03d}".format(it),
                      "train_loss:{:.4f}".format(train_loss),
                      "train_acc:{:.4f}".format(train_acc),
                      "val_loss:{:.4f}".format(val_loss),
                      "val_acc:{:.4f}".format(val_acc),
                      "time:{:.4f}".format(toc - tic))
        return train_losses

    def loss_fn(self, idx, labels, training=True):
        if training:
            # .nnz 是获得X中元素的个数
            _X = sparse_dropout(self.X_tf, self.dropout, [self.X.nnz])
        else:
            _X = self.X_tf

        self.h1 = self.layer1([self.An_tf, _X])
        if training:
            _h1 = tf.nn.dropout(self.h1, self.dropout)
        else:
            _h1 = self.h1

        self.h2 = self.layer2([self.An_tf, _h1])
        self.var_list = self.layer1.weights + self.layer2.weights
        # calculate the loss base on idx and labels
        _logits = tf.gather(self.h2, idx)
        _loss_per_node = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                 logits=_logits)
        _loss = tf.reduce_mean(_loss_per_node)
        # 加上 l2 正则化项
        _loss += FLAGS.weight_decay * sum(map(tf.nn.l2_loss, self.layer1.weights))
        return _loss

    def evaluate(self, idx, true_labels, training):
        K = true_labels.max() + 1
        _loss = self.loss_fn(idx, np.eye(K)[true_labels], training=training).numpy()
        _pred_logits = tf.gather(self.h2, idx)
        _pred_labels = tf.argmax(_pred_logits, axis=1).numpy()
        _acc = accuracy_score(_pred_labels, true_labels)
        return _loss, _acc


# In[8]:


# 计算标准化的邻接矩阵：根号D * A * 根号D
def preprocess_graph(adj):
    # _A = A + I
    _adj = adj + sp.eye(adj.shape[0])
    # _dseq：各个节点的度构成的列表
    _dseq = _adj.sum(1).A1
    # 构造开根号的度矩阵
    _D_half = sp.diags(np.power(_dseq, -0.5))
    # 计算标准化的邻接矩阵, @ 表示矩阵乘法
    adj_normalized = _D_half @ _adj @ _D_half
    return adj_normalized.tocsr()

if __name__ == "__main__":
    # 读取数据
    # A_mat：邻接矩阵
    # X_mat：特征矩阵
    # z_vec：label
    # train_idx,val_idx,test_idx: 要使用的节点序号
    A_mat, X_mat, z_vec, train_idx, val_idx, test_idx = load_data_planetoid(FLAGS.dataset)
    # 邻居矩阵标准化
    An_mat = preprocess_graph(A_mat)

    # 节点的类别个数
    K = z_vec.max() + 1

    # 构造GCN模型
    gcn = GCN(An_mat, X_mat, [FLAGS.hidden1, K])
    # 训练
    gcn.train(train_idx, z_vec[train_idx], val_idx, z_vec[val_idx])
    # 测试
    test_res = gcn.evaluate(test_idx, z_vec[test_idx], training=False)
    print("Dataset {}".format(FLAGS.dataset),
          "Test loss {:.4f}".format(test_res[0]),
          "test acc {:.4f}".format(test_res[1]))


# In[ ]:




