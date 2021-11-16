import glob
import os
import json
import logging
import math
import numpy as np
import torch
import pprint
import utils

class Dataset(object):
    """
    This module implements the APIs for loading dataset and providing batch data
    """
    def __init__(self, args):
        self.logger = logging.getLogger("GraphCM")
        self.max_d_num = args.max_d_num
        self.gpu_num = args.gpu_num
        self.dataset = args.dataset
        self.data_dir = os.path.join('data', self.dataset)
        self.args = args
        
        self.train_set = self.load_dataset(os.path.join(self.data_dir, 'train_per_query_quid.txt'), mode='train')
        self.valid_set = self.load_dataset(os.path.join(self.data_dir, 'valid_per_query_quid.txt'), mode='valid')
        self.test_set = self.load_dataset(os.path.join(self.data_dir, 'test_per_query_quid.txt'), mode='test')
        self.label_set = self.load_dataset(os.path.join(self.data_dir, 'human_label_for_GraphCM_per_query_quid.txt'), mode='label')
        self.trainset_size = len(self.train_set)
        self.validset_size = len(self.valid_set)
        self.testset_size = len(self.test_set)
        self.labelset_size = len(self.label_set)

        self.query_qid = utils.load_dict(self.data_dir, 'query_qid.dict')
        self.url_uid = utils.load_dict(self.data_dir, 'url_uid.dict')
        self.vtype_vid = utils.load_dict(self.data_dir, 'vtype_vid.dict')
        self.query_size = len(self.query_qid)
        self.doc_size = len(self.url_uid)
        self.vtype_size = len(self.vtype_vid)

        self.logger.info('Train set size: {} sessions.'.format(len(self.train_set)))
        self.logger.info('Dev set size: {} sessions.'.format(len(self.valid_set)))
        self.logger.info('Test set size: {} sessions.'.format(len(self.test_set)))
        self.logger.info('Label set size: {} sessions.'.format(len(self.label_set)))
        self.logger.info('Unique query num, including zero vector: {}'.format(self.query_size))
        self.logger.info('Unique doc num, including zero vector: {}'.format(self.doc_size))
        self.logger.info('Unique vtype num, including zero vector: {}'.format(self.vtype_size))

    def load_dataset(self, data_path, mode):
        """
        Loads the dataset
        """
        data_set = []
        lines = open(data_path).readlines()
        previous_sid = -1
        qids, uids, vids, clicks, relevances = [], [], [], [], []
        for line in lines:
            attr = line.strip().split('\t')
            sid = int(attr[0].strip())
            if previous_sid != sid:
                # a new session starts
                if previous_sid != -1:
                    assert len(uids) // 10 == len(qids)
                    assert len(vids) // 10 == len(qids)
                    assert len(relevances) // 10 == len(qids)
                    assert (len(clicks) - 1) // 10 == len(qids)
                    last_rank = 0
                    for idx, click in enumerate(clicks[1:]):
                        last_rank = idx + 1 if click else last_rank
                    relevance_start = 0
                    for idx, relevance in enumerate(relevances):
                        if relevance != -1:
                            relevance_start = idx
                            assert relevance_start % 10 == 0
                            break
                    data_set.append({'sid': previous_sid,
                                    'qids': qids,
                                    'uids': uids,
                                    'vids': vids,
                                    'clicks': clicks,
                                    'last_rank': last_rank,
                                    'relevances': relevances[relevance_start : relevance_start + 10],
                                    'relevance_start': relevance_start})
                previous_sid = sid
                qids = [int(attr[1].strip())]
                uids = json.loads(attr[2].strip())
                vids = json.loads(attr[3].strip())
                clicks = [0] + json.loads(attr[4].strip())
                relevances = json.loads(attr[5].strip()) if mode == 'label' else [0 for _ in range(self.max_d_num)]
            else:
                # the previous session continues
                qids.append(int(attr[1].strip()))
                uids = uids + json.loads(attr[2].strip())
                vids = vids + json.loads(attr[3].strip())
                clicks = clicks + json.loads(attr[4].strip())
                relevances = relevances + (json.loads(attr[5].strip()) if mode == 'label' else [0 for _ in range(self.max_d_num)])
        last_rank = 0
        for idx, click in enumerate(clicks[1:]):
            last_rank = idx + 1 if click else last_rank
        relevance_start = 0
        for idx, relevance in enumerate(relevances):
            if relevance != -1:
                relevance_start = idx
                assert relevance_start % 10 == 0
                break
        data_set.append({'sid': previous_sid,
                        'qids': qids,
                        'uids': uids,
                        'vids': vids,
                        'clicks': clicks,
                        'last_rank': last_rank,
                        'relevances': relevances[relevance_start : relevance_start + 10],
                        'relevance_start': relevance_start})
        return data_set

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                        'qids': [],
                        'uids': [],
                        'vids': [],
                        'clicks': [],
                        'last_ranks': [],
                        'relevances': [],
                        'true_clicks': [],
                        'relevance_starts': []}
        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['qids'].append(sample['qids'])
            batch_data['uids'].append(sample['uids'])
            batch_data['vids'].append(sample['vids'])
            batch_data['clicks'].append(sample['clicks'])
            batch_data['last_ranks'].append(sample['last_rank'])
            batch_data['relevances'].append(sample['relevances'])
            batch_data['true_clicks'].append(sample['clicks'][1:])
            batch_data['relevance_starts'].append(sample['relevance_start'])
        return batch_data

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        """
        Generate data batches for a specific dataset (train/valid/test/label)
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'valid':
            data = self.valid_set
        elif set_name == 'test':
            data = self.test_set
        elif set_name == 'label':
            data = self.label_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()

        # alignment for multi-gpu cases
        indices += indices[:(self.gpu_num - data_size % self.gpu_num) % self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
