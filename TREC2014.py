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
import matplotlib.pyplot as plot
from tqdm import tqdm

def generate_dict_list(args):
    punc = '\\~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    session_sid = {}
    query_qid, url_uid, vtype_vid = {'': 0}, {'': 0}, {'': 0}
    uid_description = {}
    total_click_num = 0

    print('  - {}'.format('start parsing xml file...'))
    DOMTree = xml.dom.minidom.parse(os.path.join(args.input, args.dataset))
    TREC2014 = DOMTree.documentElement
    sessions = TREC2014.getElementsByTagName('session')
        
    # generate infos_per_session
    print('  - {}'.format('generating infos_per_session...'))
    infos_per_session = []
    junk_interation_num = 0
    for session in sessions:
        info_per_session = {}
        # get the session id
        session_number = int(session.getAttribute('num'))
        if not (session_number in session_sid):
            session_sid[session_number] = len(session_sid)
        info_per_session['session_number'] = session_number
        info_per_session['sid'] = session_sid[session_number]
        # print('session: {}'.format(session_number))

        # Get topic id
        topic = int(session.getElementsByTagName('topic')[0].getAttribute('num'))
        info_per_session['topic'] = topic
        
        # Get information within a query
        interactions = session.getElementsByTagName('interaction')
        interaction_infos = []
        for interaction in interactions:
            interaction_info = {}

            # Get query/document infomation
            query = interaction.getElementsByTagName('query')[0].childNodes[0].data
            docs = interaction.getElementsByTagName('results')[0].getElementsByTagName('result')
            doc_infos = []

            # Sanity check
            if len(docs) == 0:
                print('  - {}'.format('WARNING: find a query with no docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            elif len(docs) > 10:
                # more than 10 docs is not ok. May cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with more than 10 docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            elif len(docs) < 10:
                # less than 10 docs is ok. Never cause index out-of-range in embeddings
                print('  - {}'.format('WARNING: find a query with less than 10 docs: {}'.format(query)))
                junk_interation_num += 1
                continue
            
            # Pass the sanity check, save useful information
            if not (query in query_qid):
                query_qid[query] = len(query_qid)
            interaction_info['query'] = query
            interaction_info['qid'] = query_qid[query]
            interaction_info['session'] = info_per_session['session_number']
            interaction_info['sid'] = info_per_session['sid']

            for doc_idx, doc in enumerate(docs):
                # WARNING: In case there might be junk data in TREC2014 (e.g., rank > 10),  so we use manual doc_rank here
                # NOTE: Vertical type is not provided in TREC datasets. It is now only provided in TianGong-ST.
                #       So we manually set vtype equal to 0, whose corresponding qid is 1.
                doc_rank = int(doc.getAttribute('rank'))
                doc_rank = 10 if doc_rank % 10 == 0 else doc_rank % 10
                assert 1 <= doc_rank and doc_rank <= 10
                assert doc_idx + 1 == doc_rank
                url = doc.getElementsByTagName('clueweb12id')[0].childNodes[0].data
                vtype = '0'
                if not (url in url_uid):
                    url_uid[url] = len(url_uid)
                if not (vtype in vtype_vid):
                    vtype_vid[vtype] = len(vtype_vid)
                doc_info = {}
                doc_info['rank'] = doc_rank
                doc_info['url'] = url
                doc_info['uid'] = url_uid[url]
                doc_info['vtype'] = vtype
                doc_info['vid'] = vtype_vid[vtype]
                doc_info['click'] = 0
                doc_infos.append(doc_info)
                # print('      doc ranks at {}: {}'.format(doc_rank, url))

            # Get click information if there are clicked docs
            # Maybe there are no clicks in this query
            clicks = interaction.getElementsByTagName('clicked')
            if len(clicks) > 0:
                clicks = clicks[0].getElementsByTagName('click')
                total_click_num += len(clicks)
                for click in clicks:
                    clicked_doc_rank = int(click.getElementsByTagName('rank')[0].childNodes[0].data)
                    for item in doc_infos:
                        if item['rank'] == clicked_doc_rank:
                            item['click'] = 1
                            break
                    # print('      click doc ranked at {}'.format(clicked_doc_rank))
            else:
                pass
                # print('      click nothing')
            interaction_info['docs'] = doc_infos
            interaction_info['uids'] = [doc['uid'] for doc in doc_infos]
            interaction_info['vids'] = [doc['vid'] for doc in doc_infos]
            interaction_info['clicks'] = [doc['click'] for doc in doc_infos]
            interaction_infos.append(interaction_info)
        info_per_session['interactions'] = interaction_infos
        infos_per_session.append(info_per_session)
    print('  - {}'.format('abandon {} junk interactions'.format(junk_interation_num)))

    # generate infos_per_query
    print('  - {}'.format('generating infos_per_query...'))
    infos_per_query = []
    for info_per_session in infos_per_session:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            infos_per_query.append(interaction_info)

    # save and check infos_per_session
    print('  - {}'.format('save and check infos_per_session...'))
    print('    - {}'.format('length of infos_per_session: {}'.format(len(infos_per_session))))
    # pprint.pprint(infos_per_session)
    # print('length of infos_per_session: {}'.format(len(infos_per_session)))
    save_list(args.output, 'infos_per_session.list', infos_per_session)
    list1 = load_list(args.output, 'infos_per_session.list')
    assert len(infos_per_session) == len(list1)
    for idx, item in enumerate(infos_per_session):
        assert item == list1[idx]
    
    # save and check infos_per_query
    print('  - {}'.format('save and check infos_per_query...'))
    print('    - {}'.format('length of infos_per_query: {}'.format(len(infos_per_query))))
    # pprint.pprint(infos_per_query)
    # print('length of infos_per_query: {}'.format(len(infos_per_query)))
    save_list(args.output, 'infos_per_query.list', infos_per_query)
    list2 = load_list(args.output, 'infos_per_query.list')
    assert len(infos_per_query) == len(list2)
    for idx, item in enumerate(infos_per_query):
        assert item == list2[idx]

    # save and check dictionaries
    print('  - {}'.format('save and check dictionaries...'))
    print('    - {}'.format('unique session number: {}'.format(len(session_sid))))
    print('    - {}'.format('unique query number: {}'.format(len(query_qid))))
    print('    - {}'.format('unique doc number: {}'.format(len(url_uid))))
    print('    - {}'.format('unique vtype number: {}'.format(len(vtype_vid))))
    print('    - {}'.format('total click number: {}'.format(total_click_num)))
    save_dict(args.output, 'session_sid.dict', session_sid)
    save_dict(args.output, 'query_qid.dict', query_qid)
    save_dict(args.output, 'url_uid.dict', url_uid)
    save_dict(args.output, 'vtype_vid.dict', vtype_vid)

    dict1 = load_dict(args.output, 'session_sid.dict')
    dict2 = load_dict(args.output, 'query_qid.dict')
    dict3 = load_dict(args.output, 'url_uid.dict')
    dict4 = load_dict(args.output, 'vtype_vid.dict')

    assert len(session_sid) == len(dict1)
    assert len(query_qid) == len(dict2)
    assert len(url_uid) == len(dict3)
    assert len(vtype_vid) == len(dict4)

    for key in dict1:
        assert dict1[key] == session_sid[key]
    for key in dict2:
        assert dict2[key] == query_qid[key]
    for key in dict3:
        assert dict3[key] == url_uid[key]
    for key in dict4:
        assert dict4[key] == vtype_vid[key]

    print('  - {}'.format('Done'))

def generate_train_valid_test(args):
    # load entity dictionaries
    print('  - {}'.format('loading entity dictionaries...'))
    session_sid = load_dict(args.output, 'session_sid.dict')
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')
    vtype_vid = load_dict(args.output, 'vtype_vid.dict')

    # load infos_per_session.list
    print('  - {}'.format('loading the infos_per_session...'))
    infos_per_session = load_list(args.output, 'infos_per_session.list')

    # Separate all sessions into train : valid : test by config ratio
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
    
    # split train & valid & test sessions
    print('  - {}'.format('generating train & valid & test data per session...'))
    random.seed(2333)
    random.shuffle(infos_per_session)
    train_sessions = infos_per_session[:train_valid_split]
    valid_sessions = infos_per_session[train_valid_split:valid_test_split]
    test_sessions = infos_per_session[valid_test_split:]
    assert train_session_num == len(train_sessions), 'train_session_num: {}, len(train_sessions): {}'.format(train_session_num, len(train_sessions))
    assert valid_session_num == len(valid_sessions), 'valid_session_num: {}, len(valid_sessions): {}'.format(valid_session_num, len(valid_sessions))
    assert test_session_num == len(test_sessions), 'test_session_num: {}, len(test_sessions): {}'.format(test_session_num, len(test_sessions))
    assert session_num == len(train_sessions) + len(valid_sessions) + len(test_sessions), 'session_num: {}, len(train_sessions) + len(valid_sessions) + len(test_sessions): {}'.format(session_num, len(train_sessions) + len(valid_sessions) + len(test_sessions))
    
    # generate train & valid & test queries
    print('  - {}'.format('generating train & valid & test data per queries...'))
    train_queries = []
    valid_queries = []
    test_queries = []
    for info_per_session in train_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            train_queries.append(interaction_info)
    for info_per_session in valid_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            valid_queries.append(interaction_info)
    for info_per_session in test_sessions:
        interaction_infos = info_per_session['interactions']
        for interaction_info in interaction_infos:
            test_queries.append(interaction_info)
    print('    - {}'.format('train queries: {}'.format(len(train_queries))))
    print('    - {}'.format('valid queries: {}'.format(len(valid_queries))))
    print('    - {}'.format('test queries: {}'.format(len(test_queries))))
    print('    - {}'.format('total queries: {}'.format(len(train_queries) + len(valid_queries) + len(test_queries))))
    
    # Write train/valid/test query information back to txt files
    print('  - {}'.format('writing back to txt files...'))
    print('    - {}'.format('writing into {}/train_per_query.txt'.format(args.output)))
    train_query_set, train_doc_set, train_vtype_set = generate_data_per_query(train_queries, np.arange(0, len(train_queries)), args.output, 'train_per_query')
    print('    - {}'.format('writing into {}/valid_per_query.txt'.format(args.output)))
    valid_query_set, valid_doc_set, valid_vtype_set = generate_data_per_query(valid_queries, np.arange(0, len(valid_queries)), args.output, 'valid_per_query')
    print('    - {}'.format('writing into {}/test_per_query.txt'.format(args.output)))
    test_query_set, test_doc_set, test_vtype_set = generate_data_per_query(test_queries, np.arange(0, len(test_queries)), args.output, 'test_per_query')

    # statistics for cold start
    print('  - {}'.format('Statistics for Cold Start:'))
    print('    - {}'.format('Entity in valid not in train...'))
    print('      - {}'.format('query: {}'.format(len(valid_query_set - train_query_set))))
    print('      - {}'.format('doc: {}'.format(len(valid_doc_set - train_doc_set))))
    print('      - {}'.format('vtype: {}'.format(len(valid_vtype_set - train_vtype_set))))
    print('    - {}'.format('Entity in test not in train....'))
    print('      - {}'.format('query: {}'.format(len(test_query_set - train_query_set))))
    print('      - {}'.format('doc: {}'.format(len(test_doc_set - train_doc_set))))
    print('      - {}'.format('vtype: {}'.format(len(test_vtype_set - train_vtype_set))))

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
    parser = argparse.ArgumentParser('TREC2014')
    parser.add_argument('--dataset', default='TREC2014.xml',
                        help='dataset name')
    parser.add_argument('--input', default='../dataset/TREC2014/',
                        help='input path')
    parser.add_argument('--output', default='./data/TREC2014',
                        help='output path')
    parser.add_argument('--dict_list', action='store_true',
                        help='generate dicts and lists for info_per_session/info_per_query')
    parser.add_argument('--train_valid_test_data', action='store_true',
                        help='generate train/valid/test data txt')
    parser.add_argument('--dgat', action='store_true',
                        help='construct graph for double GAT')
    parser.add_argument('--cold_start', action='store_true',
                        help='construct dataset for studying cold start problems')
    parser.add_argument('--sparsity', action='store_true',
                        help='compute sparisity for the dataset')
    parser.add_argument('--trainset_ratio', default=0.8,
                        help='ratio of the train session/query according to the total number of sessions/queries')
    parser.add_argument('--validset_ratio', default=0.1,
                        help='ratio of the valid session/query according to the total number of sessions/queries')
    args = parser.parse_args()

    if args.dict_list:
        # generate info_per_session & info_per_query
        print('===> {}'.format('generating dicts and lists...'))
        generate_dict_list(args)
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