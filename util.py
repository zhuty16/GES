import numpy as np
import scipy.sparse as sp


def get_adj_matrix(train_dict, rel_dict, num_item, alpha, beta, max_len):
    row_seq = [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]
    col_seq = [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]

    row_sem = [i for i in rel_dict for j in rel_dict[i]] + [j for i in rel_dict for j in rel_dict[i]]
    col_sem = [j for i in rel_dict for j in rel_dict[i]] + [i for i in rel_dict for j in rel_dict[i]]

    rel_matrix = sp.coo_matrix(([alpha]*len(row_seq)+[beta]*len(row_sem), (row_seq+row_sem, col_seq+col_sem)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
    row_sum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def get_train_data(train_dict, num_item, max_len):
    train_data = list()
    for u in train_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        pos_seq = np.ones([max_len], dtype=np.int32) * num_item
        neg_seq = np.ones([max_len], dtype=np.int32) * num_item
        nxt = train_dict[u][-1]
        idx = max_len - 1
        for i in reversed(train_dict[u][:-1]):
            input_seq[idx] = i
            pos_seq[idx] = nxt
            if nxt != num_item:
                neg_seq[idx] = np.random.randint(num_item)
                while neg_seq[idx] == nxt:
                #while neg_seq[idx] in train_dict[u]:
                    neg_seq[idx] = np.random.randint(num_item)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        train_data.append([input_seq, pos_seq, neg_seq])
    return train_data


def get_train_batch(train_data, batch_size):
    train_batch = list()
    np.random.shuffle(train_data)
    i = 0
    while i < len(train_data):
        train_batch.append(np.asarray(train_data[i:i+batch_size]))
        i += batch_size
    return train_batch


def get_validate_data(train_dict, validate_dict, negative_dict, num_item, max_len):
    validate_data = list()
    for u in validate_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        idx = max_len - 1
        for i in reversed(train_dict[u]):
            input_seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        item_idx = [validate_dict[u]]
        for neg in negative_dict[u]:
            item_idx.append(neg)
        validate_data.append(list(input_seq) + item_idx)
    validate_data = np.asarray(validate_data)
    return validate_data


def get_test_data(train_dict, validate_dict, test_dict, negative_dict, num_item, max_len):
    test_data = list()
    for u in test_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        idx = max_len - 1
        input_seq[idx] = validate_dict[u]
        idx -= 1
        for i in reversed(train_dict[u]):
            input_seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        item_idx = [test_dict[u]]
        for neg in negative_dict[u]:
            item_idx.append(neg)
        test_data.append(list(input_seq) + item_idx)
    test_data = np.asarray(test_data)
    return test_data


def get_feed_dict(model, batch_data, emb_dropout_rate, node_dropout_rate):
    feed_dict = dict()
    feed_dict[model.emb_dropout_rate] = emb_dropout_rate
    feed_dict[model.node_dropout_rate] = node_dropout_rate
    feed_dict[model.input_seq] = batch_data[:, 0]
    feed_dict[model.pos_seq] = batch_data[:, 1]
    feed_dict[model.neg_seq] = batch_data[:, 2]
    return feed_dict


def get_feed_dict_test(model, batch_data_test, max_len):
    feed_dict = dict()
    feed_dict[model.emb_dropout_rate] = 0.0
    feed_dict[model.node_dropout_rate] = 0.0
    feed_dict[model.input_seq] = batch_data_test[:, :max_len]
    feed_dict[model.test_item] = batch_data_test[:, max_len:]
    return feed_dict
