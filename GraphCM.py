# encoding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
from torch_geometric.nn import GATConv, RGCNConv, FastRGCNConv, NNConv
from modules import DGATLayer, ExamPredictor, RelEstimator

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
INF = 1e30
MINF = -(1e30)

class GraphCM(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(GraphCM, self).__init__()
        self.args = args
        self.dataset = dataset
        self.logger = logging.getLogger("GraphCM")

        self.gnn_layer = DGATLayer(args, query_size, doc_size, vtype_size, dataset)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.sigmoid = nn.Sigmoid()

        # Examination Predictor
        self.exam_predictor = ExamPredictor(args, query_size, doc_size, vtype_size, dataset)

        # Relevance Estimator
        self.rel_estimator = RelEstimator(args, query_size, doc_size, vtype_size, dataset)

        # Combination Layer
        if self.args.combine == 'exp_mul':
            self.lamda = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.mu = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.lamda.data.fill_(1.0)
            self.mu.data.fill_(1.0)
        elif self.args.combine == 'linear':
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0.5)
            self.beta.data.fill_(0.5)
        elif self.args.combine == 'nonlinear':
            self.w11 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w31 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w32 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)
            self.w31.data.fill_(0.5)
            self.w32.data.fill_(0.5)
    
    def combine(self, exams, rels):
        '''
        Combine examination and relevacne to get the click probability
        '''
        combine = self.args.combine
        if combine == 'mul':
            clicks = torch.mul(rels, exams)
        elif combine == 'exp_mul':
            clicks = torch.mul(torch.pow(rels, self.lamda), torch.pow(exams, self.mu))
        elif combine == 'linear':
            clicks = torch.add(torch.mul(rels, self.alpha), torch.mul(exams, self.beta))
        elif combine == 'nonlinear':  # 2-layer
            out1 = self.sigmoid(torch.add(torch.mul(rels, self.w11), torch.mul(exams, self.w12)))
            out2 = self.sigmoid(torch.add(torch.mul(rels, self.w21), torch.mul(exams, self.w22)))
            clicks = self.sigmoid(torch.add(torch.mul(out1, self.w31), torch.mul(out2, self.w32)))
        else:
            raise NotImplementedError('Unsupported combination type: {}'.format(combine))
        return clicks

    def forward(self, qids, uids, vids, clicks):
        batch_size = len(qids)
        seq_len = len(qids[0])

        # Get embeddings, which is already padded
        qid_embed, uid_embed, vid_embed, click_embed, pos_embed = self.gnn_layer(qids, uids, vids, clicks, self.args.use_gnn)
        qu_interactions = self.gnn_layer.interact_neighs(qids, uids) if self.args.inter_neigh_sample > 0 else None
        
        # Examination predition process
        exams = self.exam_predictor(vid_embed, click_embed, pos_embed)

        # Relevance estimation process
        rels = self.rel_estimator(qid_embed, uid_embed, vid_embed, click_embed, pos_embed, qu_interactions)
        
        # Combination Layer
        pred_logits = self.combine(exams, rels)

        return pred_logits, rels