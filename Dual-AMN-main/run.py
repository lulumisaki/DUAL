import warnings

warnings.filterwarnings('ignore')

import os
import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
from evaluate import evaluate
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer_new import NR_GraphAttention
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

seed = 12306
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data("data/en_fr_15k_V1/",
                                                                                         train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 2
dropout_rate = 0.0
lr = 0.005
gamma = 1
depth = 2


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


def get_trgat(node_hidden, rel_hidden, triple_size=triple_size, node_size=node_size, rel_size=rel_size, dropout_rate=0,
              gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    e_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)

    r_encoder = NR_GraphAttention(node_size, activation="tanh",
                                  rel_size=rel_size,
                                  use_bias=True,
                                  depth=depth,
                                  triple_size=triple_size)

    # out_feature = Concatenate(-1)([e_encoder([ent_feature]+opt),r_encoder([rel_feature]+opt)])
    # out_feature = Dropout(dropout_rate)(out_feature)

    out_feature = e_encoder([ent_feature] + opt)
    # out_feature = r_encoder([rel_feature]+opt)

    alignment_input = Input(shape=(None, 2))

    def align_loss(tensor):
        def squared_dist(x):
            A, B = x
            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
            return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)

        emb = tensor[1]
        l, r = K.cast(tensor[0][0, :, 0], 'int32'), K.cast(tensor[0][0, :, 1], 'int32')
        l_emb, r_emb = K.gather(reference=emb, indices=l), K.gather(reference=emb, indices=r)

        pos_dis = K.sum(K.square(l_emb - r_emb), axis=-1, keepdims=True)
        r_neg_dis = squared_dist([r_emb, emb])
        l_neg_dis = squared_dist([l_emb, emb])

        l_loss = pos_dis - l_neg_dis + gamma
        l_loss = l_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = pos_dis - r_neg_dis + gamma
        r_loss = r_loss * (
                    1 - K.one_hot(indices=l, num_classes=node_size) - K.one_hot(indices=r, num_classes=node_size))

        r_loss = (r_loss - K.stop_gradient(K.mean(r_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(r_loss, axis=-1, keepdims=True))
        l_loss = (l_loss - K.stop_gradient(K.mean(l_loss, axis=-1, keepdims=True))) / K.stop_gradient(
            K.std(l_loss, axis=-1, keepdims=True))

        lamb, tau = 30, 10
        l_loss = K.logsumexp(lamb * l_loss + tau, axis=-1)
        r_loss = K.logsumexp(lamb * r_loss + tau, axis=-1)
        return K.mean(l_loss + r_loss)

    # loss = Lambda(align_loss)([alignment_input,out_feature])

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=e_encoder.bias)
    # train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=rel_emb)

    return train_model, feature_model


debug_model, debug_model_02 = get_trgat(dropout_rate=dropout_rate,
                                        node_size=node_size,
                                        rel_size=rel_size,
                                        depth=depth,
                                        gamma=gamma,
                                        node_hidden=node_hidden,
                                        rel_hidden=rel_hidden,
                                        lr=lr)

for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in range(len(train_pair) // batch_size + 1)]:
    if len(pairs) == 0:
        continue
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, pairs]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]

    # print(pairs)
    vec = debug_model.predict_on_batch(inputs)
    print(vec.shape)

    with open("init_tensor_values/en_fr_15k_V1/e_encoder_bias.pkl",
              "wb") as fo_token_embedding:
        pickle.dump(vec, fo_token_embedding)
    exit()

    vec = debug_model.predict_on_batch(inputs[:-1])
    print("==")
    # print(vec[vec!=0])
    print(vec)
    print(vec.shape)

    # with open("RelTokenEmbedding.pkl","wb") as fo_token_embedding:
    #     pickle.dump(vec,fo_token_embedding)
    exit()