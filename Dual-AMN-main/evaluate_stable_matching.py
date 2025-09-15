import pickle as pkl
import numpy as np
from collections import defaultdict


class evaluate:
    def __init__(self, dev_pair):
        self.dev_pair = dev_pair

    def dot(self, A, B):
        A_sim = np.dot(A, B.T)
        return np.expand_dims(A_sim, axis=0)

    def avg(self, x):
        k = 10
        top_k_indices = x[0]
        top_k_indices = np.argpartition(top_k_indices, -k, axis=1)[:, -k:]
        top_k_values = np.take_along_axis(x[0], top_k_indices, axis=1)

        return np.expand_dims(np.sum(top_k_values, axis=-1) / k, axis=0)

    def stable_match_sim(self, sim, LR, RL, k=10):
        sim, LR, RL = [np.squeeze(m, axis=0) for m in [sim, LR, RL]]
        LR, RL = [np.expand_dims(m, axis=1) for m in [LR, RL]]
        sim = 2 * sim - np.transpose(LR)
        sim = sim - RL
        # rank = np.argsort(-sim, axis=-1)[:, :k]
        rank = np.argsort(-sim, axis=-1)
        return rank

    def male_without_match(self, matches, males):
        for male in males:
            if male not in matches:
                return male

    def deferred_acceptance(self, male_prefs, female_prefs):
        female_queue = defaultdict(int)
        males = list(male_prefs.keys())
        matches = {}
        while True:
            male = self.male_without_match(matches, males)
            # print(male)
            if male is None:
                break
            female_index = female_queue[male]
            female_queue[male] += 1
            # print(female_index)

            try:
                female = male_prefs[male][female_index]
            except IndexError:
                matches[male] = male
                continue
            # print('Trying %s with %s... ' % (male, female), end='')
            prev_male = matches.get(female, None)
            if not prev_male:
                matches[male] = female
                matches[female] = male
                # print('auto')
            elif female_prefs[female].index(male) < female_prefs[female].index(
                prev_male
            ):
                matches[male] = female
                matches[female] = male
                del matches[prev_male]
                # print('matched')
            # else:
            # print('rejected')
        return {male: matches[male] for male in male_prefs.keys()}

    def CSLS_cal(self, Lvec, Rvec, evaluate=True, batch_size=1024):
        L_sim, R_sim = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            L_sim.append(
                self.dot(Lvec[epoch * batch_size : (epoch + 1) * batch_size], Rvec)
            )
            R_sim.append(
                self.dot(Rvec[epoch * batch_size : (epoch + 1) * batch_size], Lvec)
            )

        LR, RL = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg(L_sim[epoch]))
            RL.append(self.avg(R_sim[epoch]))

        r_rank_idx_dict = {}
        l_rank_idx_dict = {}
        for epoch in range(len(Lvec) // batch_size + 1):
            r_sim = self.stable_match_sim(
                R_sim[epoch], np.concatenate(LR, axis=1), RL[epoch]
            )
            l_sim = self.stable_match_sim(
                L_sim[epoch], np.concatenate(RL, axis=1), LR[epoch]
            )

            for inner_idx, idx in enumerate(
                range(epoch * batch_size, (epoch * batch_size) + len(r_sim))
            ):
                r_rank_idx_dict[idx] = r_sim[inner_idx].tolist()
                l_rank_idx_dict[idx] = l_sim[inner_idx].tolist()

        matches = self.deferred_acceptance(r_rank_idx_dict, l_rank_idx_dict)

        trueC = 0
        for match in matches:
            if match == matches[match]:
                trueC += 1
        print("accuracy： " + str(float(trueC) / len(matches)))

    def test(self, Lvec, Rvec):
        self.CSLS_cal(Lvec, Rvec)

        # def cal(results):
        #     hits1, hits5, hits10, mrr = 0, 0, 0, 0
        #     for x in results[:,1]:
        #         if x < 1:
        #             hits1 += 1
        #         if x < 5:
        #             hits5 += 1
        #         if x < 10:
        #             hits10 += 1
        #         mrr += 1 / (x + 1)
        #     return hits1, hits5, hits10, mrr

        # hits1, hits5, hits10, mrr = cal(results)
        # print("Hits@1:", hits1 / len(Lvec), "Hits@5:", hits5 / len(Lvec), "Hits@10:", hits10 / len(Lvec), "MRR:", mrr / len(Lvec))
        # return results


if __name__ == "__main__":
    with open("/root/autodl-fs/Dual-AMN/lvec_rvec.pkl", "rb") as fi_lvec_rvec:
        lvec_rvec_datas = pkl.load(fi_lvec_rvec)
    with open("/root/autodl-fs/Dual-AMN/dev_pair.pkl", "rb") as fi_dev_pair:
        dev_pair_datas = pkl.load(fi_dev_pair)

    evaluate_local = Evaluate(dev_pair_datas)
    evaluate_local.test(lvec_rvec_datas[0], lvec_rvec_datas[1])

