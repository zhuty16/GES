import time
import random
import argparse
import numpy as np
import tensorflow as tf
from evaluate import evaluate
from util import *
from GES_SASRec import GES_SASRec


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Amazon')
parser.add_argument('--num_factor', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--num_block', type=int, default=1)
parser.add_argument('--num_head', type=int, default=1)
parser.add_argument('--emb_dropout_rate', type=float, default=0.1)
parser.add_argument('--node_dropout_rate', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--gnn', type=str, default='sgc') # {sgc, gcn}
parser.add_argument('--num_layer', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--layer_agg', type=str, default='none') # {none, sum. avg, concat}
args = parser.parse_args()
print(vars(args))

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    [train_dict, validate_dict, test_dict, negative_dict, num_user, num_item] = np.load('data/{dataset}/{dataset}.npy'.format(dataset=args.dataset), allow_pickle=True)
    rel_dict = np.load('data/{dataset}/{dataset}_rel.npy'.format(dataset=args.dataset), allow_pickle=True)[0]
    print('num_user:%d, num_item:%d' % (num_user, num_item))

    train_dict_len = [len(train_dict[u]) for u in train_dict]
    print('max len: %d, min len:%d, avg len:%.2f' % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len)))

    adj_matrix = get_adj_matrix(train_dict, rel_dict, num_item, args.alpha, args.beta, args.max_len)

    print('Model preparing...')
    model = GES_SASRec(adj_matrix, num_user, num_item, args)
    sess.run(tf.global_variables_initializer())
    validate_data = get_validate_data(train_dict, validate_dict, negative_dict, num_item, args.max_len)
    test_data = get_test_data(train_dict, validate_dict, test_dict, negative_dict, num_item, args.max_len)

    print('Model training...')
    for epoch in range(1, args.num_epoch+1):
        t1 = time.time()
        train_loss = list()
        train_data = get_train_data(train_dict, num_item, args.max_len)
        train_batch = get_train_batch(train_data, args.batch_size)
        for batch in train_batch:
            loss, _ = sess.run([model.loss, model.train_op], feed_dict=get_feed_dict(model, batch, args.emb_dropout_rate, args.node_dropout_rate))
            train_loss.append(loss)
        train_loss = np.mean(train_loss)
        print('epoch: %d, %.2fs' % (epoch, time.time() - t1))
        print('training loss: %.4f' % (train_loss))

        batch_size_test = 1000
        rank_list = list()
        for start in range(0, num_user, batch_size_test):
            test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_test(model, validate_data[start:start+batch_size_test], args.max_len))
            rank_list += np.reshape(test_logits, [-1, 1000]).argsort()[:, ::-1].tolist()
        metric_validate_10 = evaluate(rank_list, 0, 10)

    print('Model testing...')
    batch_size_test = 1000
    rank_list = list()
    for start in range(0, num_user, batch_size_test):
        test_logits = sess.run(model.test_logits, feed_dict=get_feed_dict_test(model, test_data[start:start+batch_size_test], args.max_len))
        rank_list += np.reshape(test_logits, [-1, 1000]).argsort()[:, ::-1].tolist()
    metric_test_10 = evaluate(rank_list, 0, 10)
