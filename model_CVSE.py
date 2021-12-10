# -----------------------------------------------------------
# Consensus-Aware Visual-Semantic Embedding implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Consensus-Aware Visual-Semantic Embedding for Image-Text Matching"
# Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma
#
# Writen by Haoran Wang, 2020
# ---------------------------------------------------------------

from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchtext
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm

from util.utils import *
from util.C_GCN import C_GCN

from cider.cider.pyciderevalcap.ciderD.ciderD import CiderD
import random




'''Image Encoder'''
def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        '''features_mean: visual initial memory'''
        features_mean = torch.mean(features, 1)

        '''choose whether to l2norm'''
        # if not self.no_imgnorm:
        #     features_mean = l2norm(features_mean)

        return features, features_mean

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


''' Text encoder'''
class EncoderText(nn.Module):
    '''This func can utilize w2v initialization for word embedding'''

    def __init__(self, wemb_type, word2idx, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bidirectional_RNN=True, no_txtnorm=False,
                 use_abs=False, RNN_type='GRU'):

        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        self.use_bidirectional_RNN = use_bidirectional_RNN
        self.RNN_type = RNN_type
        if RNN_type == 'GRU':
            self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)
        elif RNN_type == 'LSTM':
            self.rnn = nn.LSTM(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bidirectional_RNN)

        self.dropout = nn.Dropout(opt.dropout_rate)

        # self.init_weights()
        '''change here'''
        self.init_weights(wemb_type, word2idx, word_dim)


    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type.lower() == 'random_init':
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))


    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.dropout(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bidirectional_RNN:
            cap_emb = (cap_emb[:, :, : int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # take absolute value, used by order embeddings
        if self.use_abs:
            cap_emb = torch.abs(cap_emb)

        cap_emb_mean = torch.mean(cap_emb, 1)
        if not self.no_txtnorm:
            cap_emb_mean = l2norm(cap_emb_mean)

        return cap_emb, cap_emb_mean


''' Visual self-attention module '''
class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, use_bn, activation_type, dropout_rate, img_region_num):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(embed_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.BatchNorm1d(img_region_num),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2norm((v_att))

        return v_att, weights


''' Textual self-attention module '''
class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, use_bn, activation_type, dropout_rate):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """

        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2norm(u_att)

        return u_att, weights


'''Fusing instance-level feature and consensus-level feature'''
class Multi_feature_fusing(nn.Module):
    """
    Emb the features from both modalities to the joint attribute label space.
    """

    def __init__(self, embed_dim, fuse_type='weight_sum'):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Multi_feature_fusing, self).__init__()

        self.fuse_type = fuse_type
        self.embed_dim = embed_dim
        if fuse_type == 'concat':
            input_dim = int(2*embed_dim)
            self.joint_emb_v = nn.Linear(input_dim, embed_dim)
            self.joint_emb_t = nn.Linear(input_dim, embed_dim)
            self.init_weights_concat()
        if fuse_type == 'adap_sum':
            self.joint_emb_v = nn.Linear(embed_dim, 1)
            self.joint_emb_t = nn.Linear(embed_dim, 1)
            self.init_weights_adap_sum()

    def init_weights_concat(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def init_weights_adap_sum(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 1)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def forward(self, v_emb_instance, t_emb_instance, v_emb_concept, t_emb_concept):
        """
        Forward propagation.
        :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        :return: joint embbeding features for both modalities
        """
        if self.fuse_type == 'multiple':
            v_fused_emb = v_emb_instance.mul(v_emb_concept);
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = t_emb_instance.mul(t_emb_concept);
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'concat':
            v_fused_emb = torch.cat([v_emb_instance, v_emb_concept], dim=1)
            v_fused_emb = self.joint_emb_instance_v(v_fused_emb)
            v_fused_emb = l2norm(v_fused_emb)

            t_fused_emb = torch.cat([t_emb_instance, t_emb_concept], dim=1)
            t_fused_emb = self.joint_emb_instance_v(t_fused_emb)
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'adap_sum':
            v_mean = (v_emb_instance + v_emb_concept) / 2
            v_emb_instance_mat = self.joint_emb_instance_v(v_mean)
            alpha_v = F.sigmoid(v_emb_instance_mat)
            v_fused_emb = alpha_v * v_emb_instance + (1 - alpha_v) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)

            t_mean = (t_emb_instance + t_emb_concept) / 2
            t_emb_instance_mat = self.joint_emb_instance_t(t_mean)
            alpha_t = F.sigmoid(t_emb_instance_mat)
            t_fused_emb = alpha_t * t_emb_instance + (1 - alpha_t) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'weight_sum':
            alpha = 0.75
            v_fused_emb = alpha * v_emb_instance + (1 - alpha) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = alpha * t_emb_instance + (1 - alpha) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        return v_fused_emb, t_fused_emb


''' Consensus-level feature learning module '''
class Consensus_level_feature_learning(nn.Module):
    """
    Consensus-level feature learning module .
    """
    def __init__(self, image_dim, embed_dim, use_bn, activation_type, dropout_rate, attribute_num,
                 no_imgnorm=False, ):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Consensus_level_feature_learning, self).__init__()

        self.no_imgnorm = no_imgnorm
        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed attribute to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )

            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Tanh()
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.smooth_coef = 10


    def forward(self, emb_instance, concept_feature, input_modal, GT_label, GT_label_ratio):
        """
        Forward propagation.
        :param emb_instance: encoded images or text, shape: (batch_size, emb_dim)
        :param concept_feature: concept feature, shape: (att_num, emb_dim)
        :return: emb_concept: consensus-level feature
                 weights_u, weights_v: predicted concept score
        """
        W_s = self.embedding_1(concept_feature)  # (concept_num, emb_dim)

        W_v_m = self.embedding_2(emb_instance)   # (bs, emb_dim)
        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_s.size()[0], 1)   # (bs, att_num, emb_dim)

        h_s = W_s.mul(W_v_m)    # (bs, concept_num, emb_dim)

        a_s = self.embedding_3(h_s) # (bs, concept_num, 1)
        a_s = a_s.squeeze(2)        # (bs, concept_num)

        weights = self.softmax(a_s * self.smooth_coef)

        if input_modal == 'textual':

            GT_label_scale = self.softmax(GT_label * self.smooth_coef)
            weights_u = GT_label_ratio * GT_label_scale + (1 - GT_label_ratio) * weights
            concept_feature = l2norm(concept_feature)

            emb_concept = (weights_u.unsqueeze(2) * concept_feature).sum(dim=1)

            if not self.no_imgnorm:
                emb_concept = l2norm(emb_concept)
            return emb_concept, weights_u

        elif input_modal == 'visual':

            weights_v = weights
            concept_feature = l2norm(concept_feature)

            emb_concept = (weights_v.unsqueeze(2) * concept_feature).sum(dim=1)
            if not self.no_imgnorm:
                emb_concept = l2norm(emb_concept)
            return emb_concept, weights_v



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
'''
Contrastive Loss CIDEr Metric
'''
class ContrastiveLossMetric(ContrastiveLoss):

    def __init__(self, opt, margin=0, measure=False, max_violation=False):

        ContrastiveLoss.__init__(self, margin, measure, max_violation)
        self.metric_name = getattr(opt, 'metric_name', 'cider')
        self.metric_samples = getattr(opt, 'metric_samples', 'random')
        self.metric_div = getattr(opt, 'metric_div', 3.0)
        self.data_percentage = getattr(opt, 'data_percentage', 100)

        if self.metric_name == 'cider':
            if self.data_percentage == 100:
                if 'coco' in opt.data_name:
                    cached_tokens = './cider/data/coco-train-words.p'
                elif 'f30k' in opt.data_name:
                    cached_tokens = './cider/data/f30k-train-words.p'
            else:
                if 'coco' in opt.data_name:
                    cached_tokens = './cider/data/coco-train_mini_' + str(self.data_percentage) + '-words.p'
                elif 'f30k' in opt.data_name:
                    cached_tokens = './cider/data/f30k-train_mini_' + str(self.data_percentage) + '-words.p'

            self.metric = CiderD(df=cached_tokens)

        #elif self.metric_name == 'spice':
        #    self.metric = Spice()
        else:
            raise ValueError('No metric defined by that name!')


    def prepare_for_metric(self, caption_gts, caption_batch):

        res = [{'image_id': ix, 'caption': [cap]} for ix, cap in enumerate(caption_batch)]
        gts = {ix:cap for ix, cap in enumerate(caption_gts)}
        if self.metric_name == 'cider':
            return res, gts
        else: print("Not ready for Spice")
        #elif self.metric_name == 'spice':
        #    return {i['image_id']:i['caption'] for i in res}, gts

    def forward_metric(self, caption_gts, caption_batch, hard_neg_indexes):

        if type(hard_neg_indexes[0]) == torch.Tensor:
            neg_ix = (hard_neg_indexes[0].data.cpu().numpy(), hard_neg_indexes[1].data.cpu().numpy())
        else:
            neg_ix = hard_neg_indexes
        # Positive Image and Positive Caption
        res, gts = self.prepare_for_metric(caption_gts, caption_batch)
        pos_score = self.metric.compute_score(gts, res)[1]
        #if self.metric_name == 'spice':
        #    pos_score = np.array([s['All']['f'] for s in pos_score])

        # Positive Image and Negative Caption
        caption_batch = np.array(caption_batch)
        neg_samples = caption_batch[neg_ix[0]]
        # Need to convert back to list because of stupid CIDEr
        res, gts = self.prepare_for_metric(caption_gts, neg_samples.tolist())
        neg_score_i2t = self.metric.compute_score(gts, res)[1]
        #if self.metric_name == 'spice':
        #    neg_score_i2t = np.array([s['All']['f'] for s in neg_score_i2t])

        # Positive Caption and Negative Image
        neg_samples = caption_batch[neg_ix[1]]
        # Need to convert back to list because of stupid CIDEr
        res, gts = self.prepare_for_metric(caption_gts, neg_samples.tolist())
        neg_score_t2i = self.metric.compute_score(gts, res)[1]
        #if self.metric_name == 'spice':
        #    neg_score_t2i = np.array([s['All']['f'] for s in neg_score_t2i])

        return pos_score, neg_score_i2t, neg_score_t2i

    def forward(self, im, s, *args):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = scores - d1
        cost_im = scores - d2

        if self.metric_samples == 'mixed':
            metric_samples = 'hard' if random.random() < 0.5 else 'random'
        else:
            metric_samples = self.metric_samples

        if metric_samples == 'random':
            indxs = [[j for j in np.arange(im.size()[0]) if j != i] for i in range(im.size()[0])]
            hard_neg_indexes = ([random.choice(elm) for elm in indxs], [random.choice(elm) for elm in indxs])
        elif metric_samples == 'hard':
            hard_neg_indexes = (torch.topk(cost_s, 2, dim=1, largest=False)[1][:, 1],
                                torch.topk(cost_im, 2, dim=1, largest=False)[1][:, 1])
        elif metric_samples == 'soft':
            hard_neg_indexes = (torch.topk(cost_s, 2, dim=1)[1][:, 1],
                                torch.topk(cost_im, 2, dim=1)[1][:, 1])

        # Metric Calculation
        caption_gts = args[0]
        caption_batch = args[1]
        pos_score, neg_score_i2t, neg_score_t2i = self.forward_metric(caption_gts, caption_batch,
                                                                      hard_neg_indexes)

        # compare every diagonal score to scores in its column
        margin_s = torch.Tensor(pos_score - neg_score_t2i).cuda()
        margin_im = torch.Tensor(pos_score - neg_score_i2t).cuda()

        # margin_s /= torch.norm(margin_s)
        # margin_im /= torch.norm(margin_im)
        margin_s /= self.metric_div
        margin_im /= self.metric_div

        margin_s = margin_s.clamp(max=1.0)
        margin_im = margin_im.clamp(max=1.0)

        # separation of negative and positive similarity
        # pos_score = (torch.Tensor(pos_score).cuda() / self.metric_div).clamp(max=1.0)
        # neg_score_t2i = (torch.Tensor(neg_score_t2i).cuda() / self.metric_div).clamp(max=1.0)
        # neg_score_i2t = (torch.Tensor(neg_score_i2t).cuda() / self.metric_div).clamp(max=1.0)
        # cost_s = (pos_score - d1.diag()).clamp(min=0) + (scores[torch.arange(im.size(0)), hard_neg_indexes[0]] - neg_score_t2i).clamp(min=0)
        # cost_im = (pos_score - d1.diag()).clamp(min=0) + (scores[torch.arange(im.size(0)), hard_neg_indexes[1]] - neg_score_i2t).clamp(min=0)
        # caption retrieval
        cost_s = (cost_s + margin_s).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (cost_im + margin_im).clamp(min=0)
        # cost_im = (cost_im - margin_im).abs()

        # TODO: We think this is correct but if something fucks up, check this FIRST!
        cost_s = cost_s[torch.arange(im.size(0)), hard_neg_indexes[0]]
        cost_im = cost_im[torch.arange(im.size(0)), hard_neg_indexes[1]]

        return cost_s.sum() + cost_im.sum()


'''KL regularizer for softmax prob distribution'''
class KL_loss_softmax(nn.Module):
    """
    Compute KL_divergence between all prediction score (already sum=1, omit softmax function)
    """
    def __init__(self):
        super(KL_loss_softmax, self).__init__()

        self.KL_loss = nn.KLDivLoss(reduce=False)

    def forward(self, im, s):
        img_prob = torch.log(im)
        s_prob = s
        KL_loss = self.KL_loss(img_prob, s_prob)
        loss = KL_loss.sum()

        return loss


class CVSE(object):
    """
    CVSE model
    """
    def __init__(self, word2idx, opt):
        self.val = False
        self.use_metric = getattr(opt, 'use_metric', None)
        self.grad_clip = opt.grad_clip
        self.dataset_name = opt.data_name
        self.GT_label_ratio = opt.Concept_label_ratio

        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.wemb_type, word2idx, opt,
                                   opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bidirectional_RNN=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm,
                                   use_abs=opt.use_abs)

        img_region_num = 36
        # visual self-attention
        self.V_self_atten_enhance = V_single_modal_atten(opt.embed_size, opt.embed_size, opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate, img_region_num)
        # textual self-attention
        self.T_self_atten_enhance = T_single_modal_atten(opt.embed_size, opt.use_BatchNorm,
                                                         opt.activation_type, opt.dropout_rate)

        # Consensus-level feature learning module
        self.V_consensus_level_embedding = Consensus_level_feature_learning(opt.embed_size, opt.embed_size, opt.use_BatchNorm,
                                                                  opt.activation_type, opt.dropout_rate, opt.num_attribute)
        self.T_consensus_level_embedding = Consensus_level_feature_learning(opt.embed_size, opt.embed_size, opt.use_BatchNorm,
                                                                  opt.activation_type, opt.dropout_rate, opt.num_attribute)

        # Consensus_GCN
        self.C_GCN = C_GCN(opt.num_attribute, in_channel=opt.input_channel, t=0.3, adj_file=opt.adj_file,
                             norm_func=opt.norm_func_type, adj_gen_mode='Complex', opt=opt)
        # multi-level feature fusing module
        self.Multi_feature_fusing = Multi_feature_fusing(embed_dim=opt.embed_size, fuse_type=opt.feature_fuse_type)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.V_self_atten_enhance.cuda()
            self.T_self_atten_enhance.cuda()
            self.V_self_atten_enhance.cuda()
            self.T_self_atten_enhance.cuda()
            self.V_consensus_level_embedding.cuda()
            self.T_consensus_level_embedding.cuda()
            self.C_GCN.cuda()
            self.Multi_feature_fusing.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        ### 1. loss
        self.criterion_rank = ContrastiveLoss(margin=opt.margin,
                                              measure=opt.measure,
                                              max_violation=opt.max_violation)

        self.criterion_KL_softmax = KL_loss_softmax()


        ### 2. learnable parms
        params = self.get_config_optim(opt.learning_rate, opt.learning_rate_MLGCN)

        ## 3. optimizer
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        params = list(self.img_enc.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.V_self_atten_enhance.parameters())
        params += list(self.T_self_atten_enhance.parameters())
        params += list(self.V_consensus_level_embedding.parameters())
        params += list(self.T_consensus_level_embedding.parameters())
        params += list(self.Multi_feature_fusing.parameters())
        self.params = params

        self.Eiters = 0

        if self.use_metric:
            self.criterion_metric = ContrastiveLossMetric(opt)
            self.cider_weight = getattr(opt, 'cider_weight', None)



    def get_config_optim(self, lr_base, lr_MLGCN):
        return [
                {'params': self.img_enc.parameters(), 'lr': lr_base},
                {'params': self.txt_enc.parameters(), 'lr': lr_base},
                {'params': self.V_self_atten_enhance.parameters(), 'lr': lr_base},
                {'params': self.T_self_atten_enhance.parameters(), 'lr': lr_base},
                {'params': self.V_consensus_level_embedding.parameters(), 'lr': lr_base},
                {'params': self.T_consensus_level_embedding.parameters(), 'lr': lr_base},
                {'params': self.C_GCN.parameters(), 'lr': lr_MLGCN},  # C_GCN lr
                {'params': self.Multi_feature_fusing.parameters(), 'lr': lr_base}
                ]

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.V_self_atten_enhance.state_dict(),
                      self.T_self_atten_enhance.state_dict(),
                      self.V_consensus_level_embedding.state_dict(),
                      self.T_consensus_level_embedding.state_dict(),
                      self.C_GCN.state_dict(),
                      self.Multi_feature_fusing.state_dict()
                      ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.V_self_atten_enhance.load_state_dict(state_dict[2])
        self.T_self_atten_enhance.load_state_dict(state_dict[3])
        self.V_consensus_level_embedding.load_state_dict(state_dict[4])
        self.T_consensus_level_embedding.load_state_dict(state_dict[5])
        self.C_GCN.load_state_dict(state_dict[6])
        self.Multi_feature_fusing.load_state_dict(state_dict[7])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.V_self_atten_enhance.train()
        self.T_self_atten_enhance.train()
        self.V_consensus_level_embedding.train()
        self.T_consensus_level_embedding.train()
        self.C_GCN.train()
        self.Multi_feature_fusing.train()
        self.val = False

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.V_self_atten_enhance.eval()
        self.T_self_atten_enhance.eval()
        self.V_consensus_level_embedding.eval()
        self.T_consensus_level_embedding.eval()
        self.C_GCN.eval()
        self.Multi_feature_fusing.eval()
        self.val = True


    def forward_emb(self, images, captions, concept_labels, concept_input_embs, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            concept_labels = concept_labels.cuda()
            concept_input_embs = concept_input_embs.cuda()

        img_emb, img_emb_mean = self.img_enc(images)
        cap_emb, cap_emb_mean = self.txt_enc(captions, lengths)

        instance_emb_v, visual_weights = self.V_self_atten_enhance(img_emb, img_emb_mean)
        instance_emb_t, textual_weights = self.T_self_atten_enhance(cap_emb, cap_emb_mean)

        concept_basis_v = self.C_GCN(instance_emb_v, concept_input_embs, conv_mode='whole_graph')
        concept_basis_t = self.C_GCN(instance_emb_t, concept_input_embs,  conv_mode='whole_graph')

        consensus_emb_v, predict_score_v = self.V_consensus_level_embedding(instance_emb_v, concept_basis_v, input_modal='visual',
                                                                            GT_label=concept_labels, GT_label_ratio=self.GT_label_ratio)
        consensus_emb_t, predict_score_t = self.T_consensus_level_embedding(instance_emb_t, concept_basis_t, input_modal='textual',
                                                                            GT_label=concept_labels, GT_label_ratio=self.GT_label_ratio)

        fused_emb_v, fused_emb_t = self.Multi_feature_fusing(instance_emb_v, instance_emb_t,
                                                             consensus_emb_v, consensus_emb_t)

        emb_v = torch.stack((instance_emb_v, consensus_emb_v, fused_emb_v), dim=0)
        emb_t = torch.stack((instance_emb_t, consensus_emb_t, fused_emb_t), dim=0)

        return emb_v, emb_t, predict_score_v, predict_score_t


    def forward_loss(self, v_emb, t_emb, predict_score_v, predict_score_t, dataset_name, use_metric, *args):
        """Compute the loss given pairs of image and caption embeddings
        """
        if 'coco' in dataset_name or 'f30k' in dataset_name:
            # TODO: Original first weight was 3
            weight = [3, 5, 1, 2]  # loss weights
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        loss_rank = self.criterion_rank(v_emb[2], t_emb[2])
        loss_rank_instance = self.criterion_rank(v_emb[0], t_emb[0])
        loss_rank_consensus = self.criterion_rank(v_emb[1], t_emb[1])
        loss_cls_KL = self.criterion_KL_softmax(predict_score_v, predict_score_t)
        loss = weight[0] * loss_rank + weight[1] * loss_rank_instance + weight[2] * loss_rank_consensus + weight[3] * loss_cls_KL

        self.logger.update('Le_rank', loss_rank.item(), v_emb.size(0))
        self.logger.update('Le_rank_instance', loss_rank_instance.item(), v_emb.size(0))
        self.logger.update('Le_rank_consensus', loss_rank_consensus.item(), v_emb.size(0))
        self.logger.update('Le_cls_KL', loss_cls_KL.item(), v_emb.size(0))

        if use_metric and self.val == False:
            loss_cider_rank = self.cider_weight * self.criterion_metric(v_emb[2], t_emb[2], *args)
            self.logger.update('Le_Cider', loss_cider_rank.item(), v_emb.size(0))

            loss += loss_cider_rank
        return loss


    def train_emb(self, images, captions, concept_labels, concept_input_embs, lengths, ids=None, *args):

        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('GCN_lr', self.optimizer.param_groups[4]['lr'])

        # compute the embeddings
        '''! change for adding input w2v dict for GCN attribute predictor'''
        v_emb, t_emb, predict_score_v, predict_score_t = self.forward_emb(images, captions,
                                                                          concept_labels,
                                                                          concept_input_embs,
                                                                          lengths)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(v_emb, t_emb, predict_score_v, predict_score_t, self.dataset_name, self.use_metric, *args)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
