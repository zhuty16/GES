import math


def evaluate(rank_list, test_id, K):
    hit_ratio = 0
    ndcg = 0
    mrr = 0
    for line in rank_list:
        rec_list = line[:K]
        if test_id in rec_list:
            hit_ratio += 1
            ndcg += math.log(2) / math.log(rec_list.index(test_id) + 2)
            mrr += 1 / (rec_list.index(test_id) + 1)
    hit_ratio_avg = hit_ratio / len(rank_list)
    ndcg_avg = ndcg / len(rank_list)
    mrr_avg = mrr / len(rank_list)
    print('HR@{K}: %.4f, NDCG@{K}: %.4f, MRR@{K}: %.4f'.format(K=K) % (hit_ratio_avg, ndcg_avg, mrr_avg))
    return hit_ratio_avg, ndcg_avg, mrr_avg
