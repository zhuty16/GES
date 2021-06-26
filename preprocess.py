import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_count(tp, id):
    count_id = tp[[id, 'rating']].groupby(id, as_index=False)
    return count_id.size()


def filter(tp, min_user_count, min_item_count, min_timestamp, max_timestamp):
    tp = tp[tp['timestamp'] >= min_timestamp]
    tp = tp[tp['timestamp'] <= max_timestamp]

    item_count = get_count(tp, 'iid')
    tp = tp[tp['iid'].isin(item_count.index[item_count >= min_item_count])]

    user_count = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(user_count.index[user_count >= min_user_count])]

    user_count, item_count = get_count(tp, 'uid'), get_count(tp, 'iid')
    return tp, user_count, item_count


def numerize(tp, user2id, item2id):
    uid = list(map(lambda x: user2id[x], tp['uid']))
    iid = list(map(lambda x: item2id[x], tp['iid']))
    tp['uid_new'] = uid
    tp['iid_new'] = iid
    return tp


if __name__ == '__main__':
    dataset = ['Amazon', 'Yelp', 'Google'][0]

    print('data preprocessing...')
    data_path = 'data/{dataset}/ratings_{dataset}.csv'.format(dataset=dataset) # http://jmcauley.ucsd.edu/data/amazon/index_2014.html
    tp = pd.read_table(data_path, sep=',', header=None, names=['uid', 'iid', 'rating', 'timestamp'])

    MIN_USER_COUNT = 20
    MIN_ITEM_COUNT = 20

    MIN_TIMESTAMP = 1356998400
    MAX_TIMESTAMP = 1388448000

    tp, user_count, item_count = filter(tp, min_user_count=MIN_USER_COUNT, min_item_count=MIN_ITEM_COUNT, min_timestamp=MIN_TIMESTAMP, max_timestamp=MAX_TIMESTAMP)

    sparsity = float(tp.shape[0]) / user_count.shape[0] / item_count.shape[0]
    print('num_user: %d, num_item: %d, num_interaction: %d, sparsity: %.4f%%' % (user_count.shape[0], item_count.shape[0], tp.shape[0], sparsity * 100))

    plt.figure(figsize=(10, 4))
    user_count.hist(bins=100)
    plt.xlabel('number of items each user has interacted with')
    plt.show()

    plt.figure(figsize=(10, 4))
    item_count.hist(bins=100)
    plt.xlabel('number of users each item has been interacted with')
    plt.show()

    unique_uid = user_count.index
    unique_iid = item_count.index

    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    item2id = dict((iid, i) for (i, iid) in enumerate(unique_iid))

    all_tp = numerize(tp, user2id, item2id)
    all_tp.to_csv('data/{dataset}/all_data.csv'.format(dataset=dataset), index=False)

    print('data splitting...')
    all_tp_sorted = all_tp.sort_values(by=['uid_new', 'timestamp', 'iid_new'])
    users, items = np.array(all_tp_sorted['uid_new'], dtype=np.int32), np.array(all_tp_sorted['iid_new'], dtype=np.int32)
    num_user, num_item = max(users) + 1, max(items) + 1
    all_data = defaultdict(list)
    for n in range(len(users)):
        all_data[users[n]].append(items[n])

    train_data = dict()
    validate_data = dict()
    test_data = dict()
    neg_data = dict()

    random.seed(2020)
    for u in all_data:
        test_data[u] = all_data[u][-1]
        validate_data[u] = all_data[u][-2]
        train_data[u] = all_data[u][:-2]
        neg_data[u] = random.sample(list(set(range(num_item)) - set(all_data[u])), 999)

    np.save('data/{dataset}/{dataset}'.format(dataset=dataset), np.array([train_data, validate_data, test_data, neg_data, num_user, num_item]))
