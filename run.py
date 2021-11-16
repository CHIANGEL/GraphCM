# encoding:utf-8
import sys
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from model import Model
from utils import *
import pprint
from tensorboardX import SummaryWriter

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('GraphCM')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--valid', action='store_true',
                        help='perform click prediction task on valid set')
    parser.add_argument('--test', action='store_true',
                        help='perform click prediction task on test set')
    parser.add_argument('--rank', action='store_true',
                        help='perform relevance estimation task on human labeled test set')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='the number of duplicated evaluation for valid/test/rank')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adadelta',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-5,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=20000,
                                help='number of training steps')
    train_settings.add_argument('--reg_relevance', type=float, default=1.0,
                                help='regularization for relevance training')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='GraphCM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--combine', default='mul',
                                help='the combination type for examination and relevance')
    model_settings.add_argument('--embed_size', type=int, default=64,
                                help='size of the query/doc embeddings')
    model_settings.add_argument('--pos_embed_size', type=int, default=4,
                                help='size of the position embeddings')
    model_settings.add_argument('--click_embed_size', type=int, default=4,
                                help='size of the click embeddings')
    model_settings.add_argument('--vtype_embed_size', type=int, default=8,
                                help='size of the vtype embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of RNN hidden units')
    model_settings.add_argument('--max_d_num', type=int, default=10,
                                help='max number of docs in a session')
    model_settings.add_argument('--use_pretrain_embed', action='store_true',
                                help='whether use pretrained embeddings')
    model_settings.add_argument('--use_gnn', action='store_true',
                                help='whether use gnn layer')
    model_settings.add_argument('--gnn_neigh_sample', type=int, default=5,
                                help='the number of neighbors to be sampled for GAT')
    model_settings.add_argument('--gnn_att_heads', type=int, default=2,
                                help='the number of multi-head attention for GAT')
    model_settings.add_argument('--gnn_dropout', type=int, default=0,
                                help='the dropout for the gat layer')
    model_settings.add_argument('--gnn_leaky_slope', type=float, default=0.2,
                                help='leaky slope of leakyrelu for gat layer')
    model_settings.add_argument('--gnn_concat', type=bool, default=False,
                                help='whether perform concatenation in gat layer')
    model_settings.add_argument('--inter_neigh_sample', type=int, default=0,
                                help='the number of neighbor to be sampled for interaction')
    model_settings.add_argument('--inter_leaky_slope', type=float, default=0.2,
                                help='leaky slope of leakyrelu for interaction')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--dataset', default='TianGong-ST',
                                help='name of the dataset to be used')
    path_settings.add_argument('--model_dir', default='./outputs/models/',
                                help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./outputs/results/',
                                help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./outputs/summary/',
                                help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_dir', default='./outputs/log/',
                                help='path of the log file. If not set, logs are printed to console')

    path_settings.add_argument('--eval_freq', type=int, default=100,
                                help='the frequency of evaluating on the valid set when training')
    path_settings.add_argument('--check_point', type=int, default=100,
                                help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=5,
                                help='lr decay when more than the patience times of evaluation where loss/ppl do not decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                                help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                                help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                                help='gpu_num')

    return parser.parse_args()

def train(args, dataset):
    """
    trains the model
    """
    logger = logging.getLogger("GraphCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    if args.load_model > -1:
        logger.info('Reloading the model...')
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Training the model...')
    model.train(dataset)
    logger.info('Done with model training!')

def valid(args, dataset):
    """
    compute perplexity and log-likelihood for valid file
    """
    logger = logging.getLogger("GraphCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Evaluating the model on valid set...')
    summary_writer = SummaryWriter(args.summary_dir)
    sum_click_loss, sum_perplexity = 0.0, 0.0
    for i in range(args.num_iter):
        valid_batches = dataset.gen_mini_batches('valid', dataset.validset_size, shuffle=False)
        valid_click_loss, valid_rel_loss, perplexity = model.evaluate(valid_batches, dataset)
        sum_click_loss += valid_click_loss
        sum_perplexity += perplexity
        summary_writer.add_scalar('final_valid/avg_click_loss', sum_click_loss / (i + 1), i)
        summary_writer.add_scalar('final_valid/avg_perplexity', sum_perplexity / (i + 1), i)
        summary_writer.add_scalar('final_valid/click_loss', valid_click_loss, i)
        summary_writer.add_scalar('final_valid/perplexity', perplexity, i)

def test(args, dataset):
    """
    compute perplexity and log-likelihood for test file
    """
    logger = logging.getLogger("GraphCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Predict on test files...')
    summary_writer = SummaryWriter(args.summary_dir)
    sum_click_loss, sum_perplexity = 0.0, 0.0
    for i in range(args.num_iter):
        test_batches = dataset.gen_mini_batches('test', dataset.testset_size, shuffle=False)
        test_click_loss, test_rel_loss, perplexity = model.evaluate(test_batches, dataset)
        sum_click_loss += test_click_loss
        sum_perplexity += perplexity
        summary_writer.add_scalar('final_test/avg_click_loss', sum_click_loss / (i + 1), i)
        summary_writer.add_scalar('final_test/avg_perplexity', sum_perplexity / (i + 1), i)
        summary_writer.add_scalar('final_test/click_loss', test_click_loss, i)
        summary_writer.add_scalar('final_test/perplexity', perplexity, i)

def rank(args, dataset):
    """
    ranking performance on test files
    """
    logger = logging.getLogger("GraphCM")
    logger.info('Initialize the model...')
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    logger.info('model.global_step: {}'.format(model.global_step))
    assert args.load_model > -1, 'args.load_model is required to specify the model file to be loaded!'
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    logger.info('Start computing NDCG@k for ranking performance')
    trunc_levels = [1, 3, 5, 10]
    sum_ndcgs = {trunc_level: 0.0 for trunc_level in trunc_levels}
    summary_writer = SummaryWriter(args.summary_dir)
    for i in range(args.num_iter):
        label_batches = dataset.gen_mini_batches('label', dataset.labelset_size, shuffle=False)
        ndcgs = model.ranking(label_batches, dataset)
        for trunc_level in trunc_levels:
            sum_ndcgs[trunc_level] += ndcgs[trunc_level]
            summary_writer.add_scalar('final_NDCG/avg_{}'.format(trunc_level), sum_ndcgs[trunc_level] / (i + 1), i)
            summary_writer.add_scalar('final_NDCG/{}'.format(trunc_level), ndcgs[trunc_level], i)
        
def run():
    """
    Prepares and runs the whole system.
    """
    # get arguments
    args = parse_args()
    assert args.batch_size % args.gpu_num == 0
    assert args.hidden_size % 2 == 0

    # create a logger
    logger = logging.getLogger("GraphCM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    check_path(args.model_dir)
    check_path(args.result_dir)
    check_path(args.summary_dir)
    if args.log_dir:
        check_path(args.log_dir)
        file_handler = logging.FileHandler(args.log_dir + time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    logger.info('Checking the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Loading train/valid/test/label/graph data...')
    dataset = Dataset(args)
    
    if args.train:
        train(args, dataset)
    if args.valid:
        valid(args, dataset)
    if args.test:
        test(args, dataset)
    if args.rank:
        rank(args, dataset)
    logger.info('run done.')

if __name__ == '__main__':
    run()
