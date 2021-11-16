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

def generate_train_valid_test(args):
    step_pbar = tqdm(total=80779266) # There are totally 80779266 records
    query_qid, url_uid = {'': 0}, {'': 0}
    lines_quid, sid = [], 0
    print('  - {}'.format('Start parsing log information...'))
    
    for line in open(os.path.join(args.input, 'search_logs-v1.txt')):
        step_pbar.update(1)
        elements = line.strip().split()
        
        # Extract query/doc information
        query = elements[0]
        urls = elements[3:13]
        if query not in query_qid:
            query_qid[query] = len(query_qid)
        for url in urls:
            if url not in url_uid:
                url_uid[url] = len(url_uid)
        qid = query_qid[query]
        uids = [url_uid[url] for url in urls]
        
        # Extract click information
        nc = int(elements[13]) # total number of clicks within this query
        clickpos = [int(pos) for pos in elements[15::2] if pos != 'o' and pos != 's' and pos != '0' and pos != '11']
        clicks = [1 if pos in clickpos else 0 for pos in range(1, 11)]

        # Store line information
        lines_quid.append('{}\t{}\t{}\t{}\t{}\n'.format(sid, qid, uids, [1 for _ in range(len(uids))], clicks))
        sid += 1

    # Shuffle the dataset
    print('  - {}'.format('Shuffling the dataset...'))
    random.seed(2333)
    random.shuffle(lines_quid)
    assert len(lines_quid) == 80779266

    # Downsample the dataset due to memory limitation
    record_downsample = args.downsample
    print('  - {}'.format('Downsampling the dataset from {} records to {} records'.format(len(lines_quid), record_downsample)))
    lines_quid = lines_quid[:record_downsample]

    # WARNING: qid/uid in lines_quid & query_qid & url_uid need to be transformed due to dataset downsampling
    print('  - {}'.format('Transform qid/uid in lines_quid & query_qid & url_uid'))
    step_pbar = tqdm(total=len(lines_quid))
    qid_query = {query_qid[query]:query for query in query_qid}
    uid_url = {url_uid[url]:url for url in url_uid}
    query_qid, url_uid = {'': 0}, {'': 0}
    for i in range(len(lines_quid)):
        step_pbar.update(1)
        # Extract elements
        attr = lines_quid[i].strip().split('\t')
        sid = int(attr[0].strip())
        qid = int(attr[1].strip())
        uids = json.loads(attr[2].strip())
        clicks = json.loads(attr[4].strip())
        
        # Transform qid
        query = qid_query[qid]
        if query not in query_qid:
            query_qid[query] = len(query_qid)
        qid = query_qid[query]

        # Transform uid
        for uid in uids:
            url = uid_url[uid]
            if url not in url_uid:
                url_uid[url] = len(url_uid)
        uids = [url_uid[uid_url[uid]] for uid in uids]
        
        # Replacement for lines_quid
        lines_quid[i] = '{}\t{}\t{}\t{}\t{}\n'.format(sid, qid, uids, [1 for _ in range(len(uids))], clicks)
    
    print('  - {}'.format('There are {} unique queries.'.format(len(query_qid))))
    print('  - {}'.format('There are {} unique documents.'.format(len(url_uid))))

    print('  - {}'.format('Save rebuilt query_qid/url_uid back to files...'))
    save_dict(args.output, 'query_qid.dict', query_qid)
    save_dict(args.output, 'url_uid.dict', url_uid)

    # Separate all query records into train : valid : test by config ratio
    print('  - {}'.format('Separate all query records...'))
    record_num = len(lines_quid)
    train_record_num = int(record_num * args.trainset_ratio)
    valid_record_num = int(record_num * args.validset_ratio)
    test_record_num = record_num - train_record_num - valid_record_num
    train_valid_split = train_record_num
    valid_test_split = train_record_num + valid_record_num
    print('    - {}'.format('train/valid split at: {}'.format(train_valid_split)))
    print('    - {}'.format('valid/test split at: {}'.format(valid_test_split)))
    print('    - {}'.format('train records: {}'.format(train_record_num)))
    print('    - {}'.format('valid records: {}'.format(valid_record_num)))
    print('    - {}'.format('test records: {}'.format(test_record_num)))
    print('    - {}'.format('total records: {}'.format(record_num)))

    # Write files with qid/uid
    print('  - {}'.format('Write files with qid/uid'))
    train_records = lines_quid[:train_valid_split]
    valid_records = lines_quid[train_valid_split:valid_test_split]
    test_records = lines_quid[valid_test_split:]
    assert train_record_num == len(train_records), 'train_record_num: {}, len(train_records): {}'.format(train_record_num, len(train_records))
    assert valid_record_num == len(valid_records), 'valid_record_num: {}, len(valid_records): {}'.format(valid_record_num, len(valid_records))
    assert test_record_num == len(test_records), 'test_record_num: {}, len(test_records): {}'.format(test_record_num, len(test_records))
    assert record_num == len(train_records) + len(valid_records) + len(test_records), 'record_num: {}, len(train_records) + len(valid_records) + len(test_records): {}'.format(record_num, len(train_records) + len(valid_records) + len(test_records))
    print('    - {}'.format('Write train_per_query_quid.txt'))
    file = open(os.path.join(args.output, 'train_per_query_quid.txt'), 'w')
    for record in train_records:
        file.write(record)
    file.close()
    print('    - {}'.format('Write valid_per_query_quid.txt'))
    file = open(os.path.join(args.output, 'valid_per_query_quid.txt'), 'w')
    for record in valid_records:
        file.write(record)
    file.close()
    print('    - {}'.format('Write test_per_query_quid.txt'))
    file = open(os.path.join(args.output, 'test_per_query_quid.txt'), 'w')
    for record in test_records:
        file.write(record)
    file.close()

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
    for session_info in train_set:
        step_pbar.update(1)
        train_query_set = train_query_set | set(session_info['qids'])
    print('    - {}'.format('unique train query num: {}'.format(len(train_query_set))))

    # Filter and split test set into: 
    #   (1) sessions whose queries have appeared in train set
    #   (2) sessions that contain queries that do not appear in train set
    cold_query_cnt, warm_query_cnt = 0, 0
    print('  - {}'.format('Start filtering'))
    step_pbar = tqdm(total=len(test_set))
    cold_test_set, warm_test_set = [], []
    for session_info in test_set:
        step_pbar.update(1)
        is_cold = False
        for qid in session_info['qids']:
            if qid not in train_query_set:
                cold_query_cnt += 1
                is_cold = True
            else:
                warm_query_cnt += 1
        if is_cold:
            cold_test_set.append(session_info)
        else:
            warm_test_set.append(session_info)
    print('  - {}'.format('Cold session num: {}'.format(len(cold_test_set))))
    print('  - {}'.format('Warm session num: {}'.format(len(warm_test_set))))
    print('  - {}'.format('Total session num: {}'.format(len(cold_test_set) + len(warm_test_set))))
    print('  - {}'.format('Cold query num: {}'.format(cold_query_cnt)))
    print('  - {}'.format('Warm query num: {}'.format(warm_query_cnt)))
    print('  - {}'.format('Total query num: {}'.format(cold_query_cnt + warm_query_cnt)))

    # Save cold/warm test set back to files
    print('  - {}'.format('Write back cold test set'))
    file = open(os.path.join(args.output, 'cold_test_per_query_quid.txt'), 'w')
    for session_info in cold_test_set:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()
    print('  - {}'.format('Write back warm test set'))
    file = open(os.path.join(args.output, 'warm_test_per_query_quid.txt'), 'w')
    for session_info in warm_test_set:
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
    qu_set, q_set, u_set = set(), set(), set()
    for set_name in set_names:
        print('    - {}'.format('Counting the query-doc pairs in the {} set'.format(set_name)))
        lines = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(set_name))).readlines()
        for line in lines:
            attr = line.strip().split('\t')
            qid = int(attr[1].strip())
            uids = json.loads(attr[2].strip())
            for uid in uids:
                qu_set.add(str([qid, uid]))
                q_set.add(qid)
                u_set.add(uid)
    
    # Compute the sparsity
    assert len(q_set) + 1 == len(query_qid)
    assert len(u_set) + 1 == len(url_uid)
    print('  - {}'.format('There are {} unique query-doc pairs in the dataset...'.format(len(qu_set))))
    print('  - {}'.format('There are {} unique queries in the dataset...'.format(len(q_set))))
    print('  - {}'.format('There are {} unique docs in the dataset...'.format(len(u_set))))
    print('  - {}'.format('There are {} possible query-doc pairs in the whole dataset...'.format(len(q_set) * len(u_set))))
    print('  - {}'.format('The sparsity is: 1 - {} / {} = {}%'.format(len(qu_set), len(q_set) * len(u_set), 100 - 100 * len(qu_set) / (len(q_set) * len(u_set)))))

def relevance(args):
    # qu_count = {}
    # # Start counting click/nonclick
    # step_pbar = tqdm(total=80779266) # There are totally 80779266 records
    # print('  - {}'.format('Start counting click/nonclick...'))
    # for line in open(os.path.join(args.input, 'search_logs-v1.txt')):
    #     step_pbar.update(1)
    #     elements = line.strip().split()
        
    #     # Extract query/doc information
    #     query = elements[0]
    #     urls = elements[3:13]
    #     if query not in qu_count:
    #         qu_count[query] = {}
    #     for url in urls:
    #         if url not in qu_count[query]:
    #             qu_count[query][url] = {'click': 0, 'nonclick': 0}
        
    #     # Extract click information
    #     nc = int(elements[13]) # total number of clicks within this query
    #     clickpos = [int(pos) for pos in elements[15::2] if pos != 'o' and pos != 's' and pos != '0' and pos != '11']
    #     clicks = [1 if pos in clickpos else 0 for pos in range(1, 11)]

    #     # Store query-url click information
    #     for url, click in zip(urls, clicks):
    #         if click:
    #             qu_count[query][url]['click'] += 1
    #         else:
    #             qu_count[query][url]['nonclick'] += 1

    # # Start storing relevance
    # print('  - {}'.format('Start storing relevance...'))
    # step_pbar = tqdm(total=659710) # There are totally 659710 lines
    # for line in open(os.path.join(args.input, 'relevance_judgments-v1.txt')):
    #     step_pbar.update(1)
    #     elements = line.strip().split()
    #     query = elements[0]
    #     url = elements[1]
    #     rel = int(elements[2])
    #     qu_count[query][url]['rel'] = rel
    
    # # Save qu_count.dict
    # print('  - {}'.format('Save qu_count.dict...'))
    # save_dict(args.output, 'qu_count.dict', qu_count)

    qu_count = load_dict(args.output, 'qu_count.dict')
    rel2click = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    rel2ratio = {}
    for qid in qu_count:
        for uid in qu_count[qid]:
            if 'rel' in qu_count[qid][uid]:
                click = qu_count[qid][uid]['click']
                nonclick = qu_count[qid][uid]['nonclick']
                rel = qu_count[qid][uid]['rel']
                rel2click[rel].append(click / (click + nonclick))
    for rel in rel2click:
        rel2ratio[rel] = sum(rel2click[rel]) / len(rel2click[rel]) if len(rel2click[rel]) else 0
    save_dict(args.output, 'rel2click.dict', rel2click)
    save_dict(args.output, 'rel2ratio.dict', rel2ratio)
    pprint.pprint(rel2ratio)
    
def main():
    parser = argparse.ArgumentParser('Yahoo')
    parser.add_argument('--input', default='../dataset/Yahoo/',
                        help='input path')
    parser.add_argument('--output', default='./data/Yahoo',
                        help='output path')
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
    parser.add_argument('--relevance', action='store_true',
                        help='statistics for relevance')
    parser.add_argument('--trainset_ratio', default=0.8,
                        help='ratio of the train session/query according to the total number of records/queries')
    parser.add_argument('--validset_ratio', default=0.1,
                        help='ratio of the valid session/query according to the total number of records/queries')
    args = parser.parse_args()
    
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
    if args.relevance:
        # statistics for relevance
        print('===> {}'.format('statistics for relevance...'))
        relevance(args)
    print('===> {}'.format('Done.'))
    
if __name__ == '__main__':
    main()