import bert
import numpy as np

sim = bert.get_bert_embedding("data/en_de_15k_V1/ent_attr_1", "data/en_de_15k_V1/ent_attr_2")
# np.fill_diagonal(sim, 0)
max_index = np.argmax(sim, axis=1)
# indices = np.argpartition(sim, -10, axis=1)[:, -10:]
# max_10 = np.array([row[row >= np.partition(row, -10)[-10]] for row in indices])
sorted_sim = np.sort(sim, axis=1)[:, ::-1]
idx = 0
hits1 = 0
hits5 = 0
hits10 = 0
for item in max_index:
    if item == idx:
        hits1 += 1
    idx += 1

print(hits1)
