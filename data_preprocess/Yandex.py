# !/usr/bin/python
# coding: utf8

from xml.dom.minidom import parse
import xml.dom.minidom
import time
import pprint
import string
import sys
sys.path.append("..")
import argparse
import re
import os
import numpy as np
import torch
import torch.nn as nn
from utils import *
from math import log
import random
import json
import matplotlib.pyplot as plt
from itertools import groupby
from tqdm import tqdm
import copy

def generate_list_dict(args):
    step_pbar = tqdm(total=34573630) # There are totally 34,573,630 sessions
    session_sid, query_qid, url_uid, vtype_vid = {'': 0}, {'': 0}, {'': 0}, {'': 0, '1': 1}
    infos_per_session = []
    junk_click_cnt = 0
    print('  - {}'.format('Start parsing log information...'))
    
    for line in open(os.path.join(args.input, 'train.txt')):
        elements = line.strip().split('\t')
        
        # Checkout the line type
        if elements[1] == 'M':
            # New session starts
            step_pbar.update(1)
            session = elements[0]
            assert session not in session_sid
            session_sid[session] = len(session_sid)
            sid = session_sid[session]
            if len(infos_per_session) > 0:
                assert len(infos_per_session[-1]['qids']) > 0
                assert len(infos_per_session[-1]['qids']) == len(infos_per_session[-1]['uidsS'])
                assert len(infos_per_session[-1]['qids']) == len(infos_per_session[-1]['clicksS'])
            infos_per_session.append({
                'sid': sid,
                'qids': [],
                'uidsS': [],
                'clicksS': [],
            })
            
        elif elements[2] == 'Q' or elements[2] == 'T':
            assert infos_per_session[-1]['sid'] == session_sid[elements[0]]
            query = elements[4]
            query_terms = elements[5:-10]
            urls_domains = elements[-10:]
            infos_per_session[-1]['uidsS'].append([])

            # Sanity Checks 0: 
            assert len(elements) == 16
            # Sanity Checks 1: Query terms must be simply separated by comma, so that length should be one
            assert len(query_terms) == 1, 'The legnth of splited query_terms list should be one\n{}\n{}'.format(line, elements)
            # Sanity Checks 2: Elements of uids list is URL,DOMAIN, so that it should only be separated by one comma
            for url_domain in urls_domains:
                url_domain_list = url_domain.split(',')
                assert len(url_domain_list) == 2
                url = url_domain_list[0]
                if url not in url_uid:
                    url_uid[url] = len(url_uid)
                infos_per_session[-1]['uidsS'][-1].append(url_uid[url])
            
            if query not in query_qid:
                query_qid[query] = len(query_qid)
            infos_per_session[-1]['qids'].append(query_qid[query])
            infos_per_session[-1]['clicksS'].append([0] * 10)

        elif elements[2] == 'C':
            assert infos_per_session[-1]['sid'] == session_sid[elements[0]]
            clicked_uid = url_uid[elements[-1]]
            if clicked_uid not in infos_per_session[-1]['uidsS'][-1]:
                junk_click_cnt += 1
            else:
                clicked_idx = infos_per_session[-1]['uidsS'][-1].index(clicked_uid)
                infos_per_session[-1]['clicksS'][-1][clicked_idx] = 1
        else:
            print(line)
            raise NotImplementedError('Unsupported line type: {}'.format(line_type))
    print('  - {}'.format('Abandon {} junk click information'.format(junk_click_cnt)))

    # Shuffle the dataset
    print('  - {}'.format('Shuffling the dataset...'))
    random.seed(2333)
    random.shuffle(infos_per_session)
    assert len(infos_per_session) == 34573630

    # Downsample the dataset due to memory limitation
    print('  - {}'.format('Downsampling the dataset from {} sessions to {} sessions'.format(len(infos_per_session), args.downsample)))
    infos_per_session = infos_per_session[:args.downsample]

    print('  - {}'.format('Save infos_per_session.list back to files...'))
    save_list(args.output, 'infos_per_session.list', infos_per_session)

def generate_train_valid_test(args):
    # Load infos_per_session.list
    print('  - {}'.format('Loading infos_per_session.list...'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')

    # Separate all sessions into train : valid : test by config ratio
    print('  - {}'.format('Separate all sessions...'))
    session_num = len(infos_per_session)
    train_session_num = int(session_num * args.trainset_ratio)
    valid_session_num = int(session_num * args.validset_ratio)
    test_session_num = session_num - train_session_num - valid_session_num
    train_valid_split = train_session_num
    valid_test_split = train_session_num + valid_session_num
    print('    - {}'.format('train/valid split at: {}'.format(train_valid_split)))
    print('    - {}'.format('valid/test split at: {}'.format(valid_test_split)))
    print('    - {}'.format('train sessions: {}'.format(train_session_num)))
    print('    - {}'.format('valid sessions: {}'.format(valid_session_num)))
    print('    - {}'.format('test sessions: {}'.format(test_session_num)))
    print('    - {}'.format('total sessions: {}'.format(session_num)))

    # Generate train/valid/test sessions
    sessions = {}
    file_types = ['train', 'valid', 'test']
    sessions['train'] = infos_per_session[:train_valid_split]
    sessions['valid'] = infos_per_session[train_valid_split:valid_test_split]
    sessions['test'] = infos_per_session[valid_test_split:]
    assert train_session_num == len(sessions['train']), 'train_session_num: {}, len(train_sessions): {}'.format(train_session_num, len(sessions['train']))
    assert valid_session_num == len(sessions['valid']), 'valid_session_num: {}, len(valid_sessions): {}'.format(valid_session_num, len(sessions['valid']))
    assert test_session_num == len(sessions['test']), 'test_session_num: {}, len(test_sessions): {}'.format(test_session_num, len(sessions['test']))
    assert session_num == len(sessions['train']) + len(sessions['valid']) + len(sessions['test']), 'session_num: {}, len(train_sessions) + len(valid_sessions) + len(test_sessions): {}'.format(session_num, len(sessions['train']) + len(sessions['valid']) + len(sessions['test']))

    # WARNING: query_qid & url_uid should be rebuilt due to dataset downsampling
    print('  - {}'.format('Write train/valid/test sessions back to files'))
    step_pbar = tqdm(total=session_num)
    query_qid, url_uid = {'': 0}, {'': 0}
    for file_type in file_types:
        print('    - {}'.format('Write {}_per_query_quid.txt'.format(file_type)))
        file = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(file_type)), 'w')
        for info_per_session in sessions[file_type]:
            step_pbar.update(1)
            sid = info_per_session['sid']
            qids = info_per_session['qids']
            uidsS = info_per_session['uidsS']
            clicksS = info_per_session['clicksS']
            for qid, uids, clicks in zip(qids, uidsS, clicksS):
                if qid not in query_qid:
                    query_qid[qid] = len(query_qid)
                for uid in uids:
                    if uid not in url_uid:
                        url_uid[uid] = len(url_uid)
                qid_print = query_qid[qid]
                uids_print = [url_uid[uid] for uid in uids]
                file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid_print, str(uids_print), str([1] * 10), str(clicks)))

    print('  - {}'.format('Save rebuilt query_qid/url_uid back to files...'))
    save_dict(args.output, 'query_qid.dict', query_qid)
    save_dict(args.output, 'url_uid.dict', url_uid)

def construct_dgat_graph(args):
    # load entity dictionaries
    print('  - {}'.format('loading entity dictionaries...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')

    # Calc edge information for train/valid/test set
    # set_names = ['demo']
    set_names = ['train', 'valid', 'test']
    qid_edges, uid_edges = set(), set()
    qid_neighbors, uid_neighbors = {qid: set() for qid in range(len(query_qid))}, {uid: set() for uid in range(len(url_uid))}
    for set_name in set_names:
        print('  - {}'.format('Constructing relations in {} set'.format(set_name)))
        lines = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(set_name))).readlines()

        # Relation 0: Query-Query within the same session
        cur_sid = -1
        qid_set = set()
        for line in lines:
            attr = line.strip().split('\t')
            sid = int(attr[0].strip())
            qid = int(attr[1].strip())
            if cur_sid == sid:
                # query in the same session
                qid_set.add(qid)
            else:
                # session ends, start creating relations
                qid_list = list(qid_set)
                for i in range(1, len(qid_list)):
                    qid_edges.add(str([qid_list[i], qid_list[i - 1]]))
                    qid_edges.add(str([qid_list[i - 1], qid_list[i]]))
                # new session starts
                cur_sid = sid
                qid_set.clear()
                qid_set.add(qid)
        # The last session
        qid_list = list(qid_set)
        for i in range(1, len(qid_list)):
            qid_edges.add(str([qid_list[i], qid_list[i - 1]]))
            qid_edges.add(str([qid_list[i - 1], qid_list[i]]))

        # Relation 1 & 2: Document of is clicked in a Query
        for line in lines:
            attr = line.strip().split('\t')
            qid = int(attr[1].strip())
            uids = json.loads(attr[2].strip())
            clicks = json.loads(attr[4].strip())
            for uid, click in zip(uids, clicks):
                if click:
                    if set_name == 'train' or set_name == 'demo':
                        qid_neighbors[qid].add(uid)
                        uid_neighbors[uid].add(qid)
        
        # Relation 3: successive Documents in the same query
        for line in lines:
            attr = line.strip().split('\t')
            uids = json.loads(attr[2].strip())
            for i in range(1, len(uids)):
                uid_edges.add(str([uids[i], uids[i - 1]]))
                uid_edges.add(str([uids[i - 1], uids[i]]))
    
    # Meta-path to q-q & u-u
    for qid in qid_neighbors:
        qid_neigh = list(qid_neighbors[qid])
        for i in range(len(qid_neigh)):
            for j in range(i + 1, len(qid_neigh)):
                uid_edges.add(str([qid_neigh[i], qid_neigh[j]]))
                uid_edges.add(str([qid_neigh[j], qid_neigh[i]]))
    for uid in uid_neighbors:
        uid_neigh = list(uid_neighbors[uid])
        for i in range(len(uid_neigh)):
            for j in range(i + 1, len(uid_neigh)):
                qid_edges.add(str([uid_neigh[i], uid_neigh[j]]))
                qid_edges.add(str([uid_neigh[j], uid_neigh[i]]))
    
    # Add self-loop
    for qid in range(len(query_qid)):
        qid_edges.add(str([qid, qid]))
    for uid in range(len(url_uid)):
        uid_edges.add(str([uid, uid]))

    # Convert & save edges information from set/list into tensor
    qid_edges = [eval(edge) for edge in qid_edges]
    uid_edges = [eval(edge) for edge in uid_edges]
    # print(qid_edges)
    # print(uid_edges)
    qid_edge_index = torch.transpose(torch.from_numpy(np.array(qid_edges, dtype=np.int64)), 0, 1)
    uid_edge_index = torch.transpose(torch.from_numpy(np.array(uid_edges, dtype=np.int64)), 0, 1)
    torch.save(qid_edge_index, os.path.join(args.output, 'dgat_qid_edge_index.pth'))
    torch.save(uid_edge_index, os.path.join(args.output, 'dgat_uid_edge_index.pth'))

    # Count degrees of qid/uid nodes
    qid_degrees, uid_degrees = [set([i]) for i in range(len(query_qid))], [set([i]) for i in range(len(url_uid))]
    for qid_edge in qid_edges:
        qid_degrees[qid_edge[0]].add(qid_edge[1])
        qid_degrees[qid_edge[1]].add(qid_edge[0])
    for uid_edge in uid_edges:
        uid_degrees[uid_edge[0]].add(uid_edge[1])
        uid_degrees[uid_edge[1]].add(uid_edge[0])
    qid_degrees = [len(d_set) for d_set in qid_degrees]
    uid_degrees = [len(d_set) for d_set in uid_degrees]
    non_isolated_qid_cnt = sum([1 if qid_degree > 1 else 0 for qid_degree in qid_degrees])
    non_isolated_uid_cnt = sum([1 if uid_degree > 1 else 0 for uid_degree in uid_degrees])
    print('  - {}'.format('Mean/Max/Min qid degree: {}, {}, {}'.format(sum(qid_degrees) / len(qid_degrees), max(qid_degrees), min(qid_degrees))))
    print('  - {}'.format('Mean/Max/Min uid degree: {}, {}, {}'.format(sum(uid_degrees) / len(uid_degrees), max(uid_degrees), min(uid_degrees))))
    print('  - {}'.format('Non-isolated qid node num: {}'.format(non_isolated_qid_cnt)))
    print('  - {}'.format('Non-isolated uid node num: {}'.format(non_isolated_uid_cnt)))

    # Save direct uid-uid neighbors for neighbor feature interactions
    uid_num = len(url_uid)
    max_node_degree = 64
    uid_neigh = [set([i]) for i in range(uid_num)]
    uid_neigh_sampler = nn.Embedding(uid_num, max_node_degree)
    for edge in uid_edges:
        src, dst = edge[0], edge[1]
        uid_neigh[src].add(dst)
        uid_neigh[dst].add(src)
    for idx, adj in enumerate(uid_neigh):
        adj_list = list(adj)
        if len(adj_list) >= max_node_degree:
            adj_sample = torch.from_numpy(np.array(random.sample(adj_list, max_node_degree), dtype=np.int64))
        else:
            adj_sample = torch.from_numpy(np.array(random.choices(adj_list, k=max_node_degree), dtype=np.int64))
        uid_neigh_sampler.weight.data[idx] = adj_sample.clone()
    torch.save(uid_neigh_sampler, os.path.join(args.output, 'dgat_uid_neighbors.pth'))

def generate_dataset_for_cold_start(args):
    def load_dataset(data_path):
        """
        Loads the dataset
        """
        data_set = []
        lines = open(data_path).readlines()
        previous_sid = -1
        qids, uids, vids, clicks = [], [], [], []
        for line in lines:
            attr = line.strip().split('\t')
            sid = int(attr[0].strip())
            if previous_sid != sid:
                # a new session starts
                if previous_sid != -1:
                    assert len(uids) == len(qids)
                    assert len(vids) == len(qids)
                    assert len(clicks) == len(qids)
                    assert len(vids[0]) == 10
                    assert len(uids[0]) == 10
                    assert len(clicks[0]) == 10
                    data_set.append({'sid': previous_sid,
                                    'qids': qids,
                                    'uids': uids,
                                    'vids': vids,
                                    'clicks': clicks})
                previous_sid = sid
                qids = [int(attr[1].strip())]
                uids = [json.loads(attr[2].strip())]
                vids = [json.loads(attr[3].strip())]
                clicks = [json.loads(attr[4].strip())]
            else:
                # the previous session continues
                qids.append(int(attr[1].strip()))
                uids.append(json.loads(attr[2].strip()))
                vids.append(json.loads(attr[3].strip()))
                clicks.append(json.loads(attr[4].strip()))
        data_set.append({'sid': previous_sid,
                        'qids': qids,
                        'uids': uids,
                        'vids': vids,
                        'clicks': clicks,})
        return data_set
    
    # Load original train/test dataset
    print('  - {}'.format('start loading train/test set...'))
    train_set = load_dataset(os.path.join(args.output, 'train_per_query_quid.txt'))
    test_set = load_dataset(os.path.join(args.output, 'test_per_query_quid.txt'))
    print('    - {}'.format('train session num: {}'.format(len(train_set))))
    print('    - {}'.format('test session num: {}'.format(len(test_set))))

    # Construct train query set for filtering
    print('  - {}'.format('Constructing train query set for filtering'))
    step_pbar = tqdm(total=len(train_set))
    train_query_set = set()
    train_doc_set = set()
    for session_info in train_set:
        step_pbar.update(1)
        train_query_set = train_query_set | set(session_info['qids'])
        for uids in session_info['uids']:
            train_doc_set = train_doc_set | set(uids)
    print('    - {}'.format('unique train query num: {}'.format(len(train_query_set))))
    print('    - {}'.format('unique train doc num: {}'.format(len(train_doc_set))))

    # Divide the full test set into four mutually exclusive parts
    print('  - {}'.format('Start the full test set division'))
    step_pbar = tqdm(total=len(test_set))
    cold_q, cold_d, cold_qd, warm_qd = [], [], [], []
    for session_info in test_set:
        step_pbar.update(1)
        is_q_cold, is_d_cold = False, False
        for qid in session_info['qids']:
            if qid not in train_query_set:
                is_q_cold = True
                break
        for uids in session_info['uids']:
            for uid in uids:
                if uid not in train_doc_set:
                    is_d_cold = True
                    break
            if is_d_cold:
                break
        if is_q_cold:
            if is_d_cold:
                cold_qd.append(session_info)
            else:
                cold_q.append(session_info)
        else:
            if is_d_cold:
                cold_d.append(session_info)
            else:
                warm_qd.append(session_info)
    print('    - {}'.format('Total session num: {}'.format(len(cold_q) + len(cold_d) + len(cold_qd) + len(warm_qd))))
    print('    - {}'.format('Cold Q session num: {}'.format(len(cold_q))))
    print('    - {}'.format('Cold D session num: {}'.format(len(cold_d))))
    print('    - {}'.format('Cold QD session num: {}'.format(len(cold_qd))))
    print('    - {}'.format('Warm QD session num: {}'.format(len(warm_qd))))

    # Save the four session sets back to files
    print('    - {}'.format('Write back cold_q set'))
    file = open(os.path.join(args.output, 'cold_q_test_per_query_quid.txt'), 'w')
    for session_info in cold_q:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()
    print('    - {}'.format('Write back cold_d set'))
    file = open(os.path.join(args.output, 'cold_d_test_per_query_quid.txt'), 'w')
    for session_info in cold_d:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()
    print('    - {}'.format('Write back cold_qd set'))
    file = open(os.path.join(args.output, 'cold_qd_test_per_query_quid.txt'), 'w')
    for session_info in cold_qd:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()
    print('    - {}'.format('Write back warm_qd set'))
    file = open(os.path.join(args.output, 'warm_qd_test_per_query_quid.txt'), 'w')
    for session_info in warm_qd:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()

def compute_sparsity(args):
    # load entity dictionaries
    print('  - {}'.format('Loading entity dictionaries...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')

    # Calc sparisity for the dataset
    # Count the query-doc pairs in the dataset
    print('  - {}'.format('Count the query-doc pairs in the dataset...'))
    # set_names = ['demo']
    set_names = ['train', 'valid', 'test']
    train_qu_set, q_set, u_set = set(), set(), set()
    for set_name in set_names:
        print('    - {}'.format('Counting the query-doc pairs in the {} set'.format(set_name)))
        lines = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(set_name))).readlines()
        for line in lines:
            attr = line.strip().split('\t')
            qid = int(attr[1].strip())
            uids = json.loads(attr[2].strip())
            for uid in uids:
                if set_name == 'train':
                    train_qu_set.add(str([qid, uid]))
                q_set.add(qid)
                u_set.add(uid)
    
    # Compute the sparsity
    assert len(q_set) + 1 == len(query_qid)
    assert len(u_set) + 1 == len(url_uid)
    print('  - {}'.format('There are {} unique query-doc pairs in the training dataset...'.format(len(train_qu_set))))
    print('  - {}'.format('There are {} unique queries in the dataset...'.format(len(q_set))))
    print('  - {}'.format('There are {} unique docs in the dataset...'.format(len(u_set))))
    print('  - {}'.format('There are {} possible query-doc pairs in the whole dataset...'.format(len(q_set) * len(u_set))))
    print('  - {}'.format('The sparsity is: 1 - {} / {} = {}%'.format(len(train_qu_set), len(q_set) * len(u_set), 100 - 100 * len(train_qu_set) / (len(q_set) * len(u_set)))))

def main():
    parser = argparse.ArgumentParser('Yandex')
    parser.add_argument('--input', default='../dataset/Yandex/',
                        help='input path')
    parser.add_argument('--output', default='./data/Yandex',
                        help='output path')
    parser.add_argument('--list_dict', action='store_true',
                        help='generate list & dict files')
    parser.add_argument('--train_valid_test_data', action='store_true',
                        help='generate train/valid/test data txt')
    parser.add_argument('--dgat', action='store_true',
                        help='construct graph for double GAT')
    parser.add_argument('--cold_start', action='store_true',
                        help='construct dataset for studying cold start problems')
    parser.add_argument('--downsample', type=int, default=10000000,
                        help='construct graph for double GAT')
    parser.add_argument('--sparsity', action='store_true',
                        help='compute sparisity for the dataset')
    parser.add_argument('--trainset_ratio', default=0.8,
                        help='ratio of the train session/query according to the total number of sessions')
    parser.add_argument('--validset_ratio', default=0.1,
                        help='ratio of the valid session/query according to the total number of sessions')
    args = parser.parse_args()
    
    if args.list_dict:
        # generate list & dict files
        print('===> {}'.format('generating train & valid & test data txt...'))
        generate_list_dict(args)
    if args.train_valid_test_data:
        # load lists saved by generate_dict_list() and generates train.txt & valid.txt & test.txt
        print('===> {}'.format('generating train & valid & test data txt...'))
        generate_train_valid_test(args)
    if args.dgat:
        # construct graph for double GAT
        print('===> {}'.format('generating graph for double GAT...'))
        construct_dgat_graph(args)
    if args.cold_start:
        # construct dataset for studying cold start problems
        print('===> {}'.format('generating dataset for studying cold start problems...'))
        generate_dataset_for_cold_start(args)
    if args.sparsity:
        # compute sparisity for the dataset
        print('===> {}'.format('compute sparisity for the dataset...'))
        compute_sparsity(args)
    print('===> {}'.format('Done.'))
    
if __name__ == '__main__':
    main()