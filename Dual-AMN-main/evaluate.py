import keras
import numpy as np
import numba as nb
from utils import *
from tqdm import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
import bert

class evaluate:
    def __init__(self,dev_pair):
        self.dev_pair = dev_pair
        
        Matrix_A = Input(shape=(None,None))
        Matrix_B = Input(shape=(None,None))

        def dot(tensor):
            k = 10
            A,B = [K.squeeze(matrix,axis=0) for matrix in tensor]
            A_sim = K.dot(A,K.transpose(B))
            return K.expand_dims(A_sim,axis=0)

        results = Lambda(dot)([Matrix_A,Matrix_B])
        self.sim_model = keras.Model(inputs = [Matrix_A,Matrix_B],outputs = results)

        k = 10
        Matrix_A = Input(shape=(None,None))
        results = Lambda(lambda x: K.expand_dims(K.sum(tf.nn.top_k(K.squeeze(x,axis=0),k=k)[0],axis=-1) / k,axis=0))(Matrix_A)
        self.avg_model = keras.Model(inputs = [Matrix_A],outputs = results)

        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=( None,))
        Ans_input = Input(shape=(None,))
        def CSLS(tensor):
            sim,LR,RL,_ = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL = [K.expand_dims(m,axis=1) for m in [LR,RL]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            return K.expand_dims(rank[:,0],axis=0)

        rank = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.rank_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = rank)

        Matrix_A = Input(shape=(None,None))
        LR_input = Input(shape=(None,))
        RL_input = Input(shape=(None,))
        Ans_input = Input(shape=(None,))

        def CSLS(tensor):
            sim,LR,RL,ans_rank = [K.squeeze(m,axis=0) for m in tensor]
            LR,RL,ans_rank = [K.expand_dims(m,axis=1) for m in [LR,RL,ans_rank]]
            sim = 2*sim - K.transpose(LR)
            sim = sim - RL
            rank = tf.argsort(-sim,axis=-1)
            results = tf.where(tf.equal(rank,K.cast(K.tile(ans_rank,[1,len(self.dev_pair)]),dtype="int32")))
            return K.expand_dims(results,axis=0)

        results = Lambda(CSLS)([Matrix_A,LR_input,RL_input,Ans_input])
        self.CSLS_model = keras.Model(inputs = [Matrix_A,LR_input,RL_input,Ans_input],outputs = results)

    def CSLS_cal(self, Lvec, Rvec, lit_sim, evaluate = True, batch_size = 1024):
        L_sim,R_sim = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(self.sim_model.predict([np.expand_dims(Lvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Rvec,axis=0)]))
            R_sim.append(self.sim_model.predict([np.expand_dims(Rvec[epoch * batch_size:(epoch + 1) * batch_size],axis=0),np.expand_dims(Lvec,axis=0)]))
        L_lit,R_lit =[],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            L_lit.append([np.expand_dims(lit_sim[epoch * batch_size:(epoch + 1) * batch_size],axis=0)])
            R_lit.append([np.expand_dims(lit_sim.T[epoch * batch_size:(epoch + 1) * batch_size],axis=0)])
        # for i in range(len(L_sim)):
        #     L_sim[i] = np.squeeze(L_sim[i] + L_lit[i], axis=0)
        #     R_sim[i] = np.squeeze(R_sim[i] + R_lit[i], axis=0)

        LR,RL = [],[]
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_model.predict([L_sim[epoch]]))
            RL.append(self.avg_model.predict([R_sim[epoch]]))

        if evaluate:
            results = []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                result = self.CSLS_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)])
                results.append(result)
            return np.concatenate(results,axis=1)[0]
        else:
            l_rank,r_rank = [],[]
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size,min((epoch+1) * batch_size,len(Lvec)))])
                r_rank.append(self.rank_model.predict([R_sim[epoch],np.concatenate(LR,axis=1),RL[epoch],np.expand_dims(ans_rank,axis=0)]))
                l_rank.append(self.rank_model.predict([L_sim[epoch],np.concatenate(RL,axis=1),LR[epoch],np.expand_dims(ans_rank,axis=0)]))

            return np.concatenate(r_rank,axis=1)[0],np.concatenate(l_rank,axis=1)[0] 

    def test(self, Lvec, Rvec, lit_sim):
        results  = self.CSLS_cal(Lvec, Rvec, lit_sim)
        train_pair, dev_pair, adj_indice, r_index, r_val, adj_features, rel_features = load_data("data/dbp_yg_15k_V1/",train_ratio=0.30)
        adj_indice = np.stack(adj_indice.nonzero(), axis=1)
        # adj_indice = tf.convert_to_tensor(adj_indice,dtype=tf.int64)
        self_nodes_idx = adj_indice[:, 0]
        self_nodes_counts = np.bincount(self_nodes_idx)
        self_nodes_idx = np.where(self_nodes_counts == 1)[0]
        def cal(results):
            hits1,hits5,hits10,mrr = 0,0,0,0
            for e,x in results:
                if e in self_nodes_idx:
                    if x < 1:
                        hits1 += 1
                    if x < 5:
                        hits5 += 1
                    if x < 10:
                        hits10 += 1
                    mrr += 1/(x + 1)
            return hits1,hits5,hits10,mrr
        hits1,hits5,hits10,mrr = cal(results)
        print("Hits@1: ",hits1/len(self_nodes_idx)," ","Hits@5: ",hits5/len(self_nodes_idx)," ","Hits@10: ",hits10/len(self_nodes_idx)," ","MRR: ",mrr/len(self_nodes_idx))
        return results