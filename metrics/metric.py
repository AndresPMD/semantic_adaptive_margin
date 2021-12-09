import argparse
import tqdm
import json
import os
import math

import numpy as np
import pandas as pd

from collections import defaultdict


class Metric:
    def __init__(self, metric, sims, recall_type, score, metric_name, recall_thresholds=[1,5,10], threshold=1,
                 dataset='coco', include_anns=False, model_name='None'):

        assert(type(score) is list)
        assert(type(recall_thresholds) is list)

        self.IMG_THRESHOLD = int(threshold)
        self.FUNCTION_MAP = {'hard': self.hard, 'soft': self.soft, 'softer': self.softer}
        self.TEXT_PER_IMG = 5
        self.RECALL_THRESHOLDS = recall_thresholds
        self.TOP_K = self.RECALL_THRESHOLDS[-1]

        self.metric_name = metric_name
        self.metric = metric
        self.sims = sims
        self.dataset = dataset
        self.include_anns = include_anns
        self.recall_type = recall_type
        self.score = score
        self.model_name = model_name

        self.intersection = []
        self.get_intersection()

    def set_sims(self, sims):
        self.sims = sims

    def get_intersection(self):

        for ix in tqdm.tqdm(range(len(self.sims)), leave=False):
            # GET MOST RELEVANT NON-GROUND TRUTH ITEMS
            intersection = []
            for i in range(self.TEXT_PER_IMG):
                index = self.TEXT_PER_IMG * ix + i
                idx = np.argsort(self.metric[index])[::-1]
                intersection.append({'indexes': idx[:self.TOP_K], 'scores': self.metric[index, idx[:self.TOP_K]]})

            count = defaultdict(int)
            for elm in intersection:
                for elm_ix, (spice_ix, sc) in enumerate(zip(elm['indexes'], elm['scores'])):
                    # count[spice_ix] += sc * (len(elm['indexes']) - elm_ix)
                    count[spice_ix] += sc

            new_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
            pop_ix = [i for i, j in enumerate(new_count) if j[0] == ix][0]
            if not self.include_anns:
                new_count.pop(pop_ix)
            self.intersection.append(new_count)

    def build_ranks(self, sims):
        ranks = {}
        for sc in self.score:
            if sc == 'softer':
                ranks[sc] = np.zeros((len(sims), self.TOP_K))
            else:
                ranks[sc] = []
        return ranks

    def calculate_ranks(self, ranks, score_type, gt_ranks=None, modality='i2t'):
        ranks = np.array(ranks)
        scores = {}
        print_str = "{} score with {}".format(score_type.capitalize(), self.recall_type.capitalize())
        # TODO: THERE IS A BUG; when IMG_THRESHOLD=1, 'hard', 'recall', 't2i'

        # Define the relevant items for real recall
        if modality == 'i2t':
            # This constant is the amount of relevant items
            num_relevant = self.TEXT_PER_IMG * self.IMG_THRESHOLD
        elif modality == 't2i':
            num_relevant = self.IMG_THRESHOLD

        # Start calculating according to score_type
        if score_type == 'hard' and self.recall_type == 'recall' and len(ranks.shape) > 1:
            for thr in self.RECALL_THRESHOLDS:
                r_at_thr = 100.0 * sum([sum(r[:thr]) / num_relevant for r in ranks]) / len(ranks)
                scores[thr] = r_at_thr
                print_str += ", R@{}: {}".format(thr, r_at_thr)

        elif score_type == 'hard':
            for thr in self.RECALL_THRESHOLDS:
                r_at_thr = 100.0 * len(np.where(ranks < thr)[0]) / len(ranks)
                scores[thr] = r_at_thr
                print_str += ", R@{}: {}".format(thr, r_at_thr)

        elif score_type == 'soft':
            for thr in self.RECALL_THRESHOLDS:
                r_at_thr = 100.0 * sum([sum(r[:thr])/num_relevant for r in ranks]) / len(ranks)
                scores[thr] = r_at_thr
                print_str += ", R@{}: {}".format(thr, r_at_thr)

        elif score_type == 'softer':
            scores['softer_order'] = {}
            for ix, thr in enumerate(self.RECALL_THRESHOLDS):
                # r_at_thr = 100.0 * ranks[:, :thr].mean(axis=1).mean(axis=0) / (
                #     gt_ranks[:, :thr].mean(axis=1).mean(axis=0))
                r_at_thr = [ranks[i, :thr].mean() / (gt_ranks[i, :thr].mean() + 1e-10) for i in range(gt_ranks.shape[0])]
                r_at_thr = 100 * np.array(r_at_thr).mean(axis=0)
                scores[thr] = r_at_thr
                print_str += ", R@{}: {}".format(thr, r_at_thr)

                # For calculation of softer score with taken into account the order of the element
                r_at_thr_order = np.array([[elm * math.log(thr - ix + 1, 2) for ix, elm in enumerate(r)]
                                           for r in ranks[:, :thr]])
                gt_ranks_at_thr_order = np.array([[elm*math.log(thr-ix+1, 2) for ix, elm in enumerate(r)]
                                                  for r in gt_ranks[:, :thr]])

                softer_order_score = [ret.mean()/(gt.mean()+1e-10)
                                                       for ret, gt in zip(r_at_thr_order, gt_ranks_at_thr_order)]
                softer_order_score = np.array(softer_order_score).mean()
                scores['softer_order'][thr] = 100.0 * softer_order_score

            print("Softer order score with {}:".format(self.recall_type.capitalize())+ ' '.join([" R@{}: {}".format(thr, sc) for thr, sc in scores['softer_order'].items()]))
        print(print_str)
        return scores

    def recall(self, ix, modality):
        if modality == 'i2t':
            # TODO: Can be optimized
            relevant_items = self.intersection[ix][:self.IMG_THRESHOLD]
            relevant_indexes = []
            for item in relevant_items:
                relevant_indexes.extend(list(range(item[0] * self.TEXT_PER_IMG,
                                                   item[0] * self.TEXT_PER_IMG + self.TEXT_PER_IMG)))
        elif modality == 't2i':
            relevant_items = self.intersection[ix // self.TEXT_PER_IMG][:self.IMG_THRESHOLD]
            relevant_indexes = [item[0] for item in relevant_items]

        return relevant_indexes

    def i2t(self):
        ranks = self.build_ranks(self.sims)
        gt_ranks = np.zeros((len(self.sims), self.TOP_K))

        for ix, sim in enumerate(tqdm.tqdm(self.sims, leave=False)):
            inds = np.argsort(sim)[::-1]

            if not self.include_anns:
                # Remove the index from the similarity
                gt = list(range(self.TEXT_PER_IMG * ix, self.TEXT_PER_IMG * ix + self.TEXT_PER_IMG, 1))
                # 100x faster
                inds = inds[~np.isin(inds, gt)]
                # More readable
                # inds = np.array([i for i in inds if i not in gt])

            for sc in self.score:
                self.FUNCTION_MAP[sc](ix, inds, ranks[sc], 'i2t', gt_ranks)

        scores = {}
        for sc in self.score:
            scores[sc] = self.calculate_ranks(ranks[sc], sc, gt_ranks, modality='i2t')

        return scores

    def t2i(self):
        sims = np.array(self.sims).T
        ranks = self.build_ranks(sims)
        gt_ranks = np.zeros((len(sims), self.TOP_K))

        for ix, sim in enumerate(tqdm.tqdm(sims, leave=False)):
            inds = np.argsort(sim)[::-1]
            if not self.include_anns:
                inds = inds[~np.isin(inds, [ix // self.TEXT_PER_IMG])]
                # inds = np.array([i for i in inds if i != ix // self.TEXT_PER_IMG])
            for sc in self.score:
                self.FUNCTION_MAP[sc](ix, inds, ranks[sc], 't2i', gt_ranks)

        scores = {}
        for sc in self.score:
            scores[sc] = self.calculate_ranks(ranks[sc], sc, gt_ranks, modality='t2i')

        return scores

    def hard(self, ix, inds, ranks, modality='i2t', gt=None):
        if modality == 'i2t':
            if self.recall_type == 'vse_recall':
                rank = 1e20
                for c in self.intersection[ix][:self.IMG_THRESHOLD]:
                    for i in range(self.TEXT_PER_IMG * c[0], self.TEXT_PER_IMG * c[0] + self.TEXT_PER_IMG, 1):
                        tmp = np.where(inds == i)[0][0]
                        if tmp < rank:
                            rank = tmp

                ranks.append(rank)

            elif self.recall_type == 'recall':
                relevant_indexes = self.recall(ix, modality)
                rel = [1 if i in relevant_indexes else 0 for i in inds[:self.TOP_K]]
                ranks.append(rel)

        elif modality == 't2i':
            if self.recall_type == 'vse_recall' or self.IMG_THRESHOLD == 1:
                rank = 1e20
                for c in self.intersection[ix // self.TEXT_PER_IMG][:self.IMG_THRESHOLD]:
                    tmp = np.where(inds == c[0])[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks.append(rank)

            elif self.recall_type == 'recall' and self.IMG_THRESHOLD >= 2:
                relevant_indexes = self.recall(ix, modality)
                rel = [1 if i in relevant_indexes else 0 for i in inds[:self.TOP_K]]
                ranks.append(rel)

    def soft(self, ix, inds, ranks, modality='i2t', gt=None):

        relevant_indexes = self.recall(ix, modality)

        if modality == 'i2t':
            # TODO: Check if correct
            constant = sum(self.metric[relevant_indexes, ix]) + 1e-20
            rel = [self.metric[i, ix] / constant if i in relevant_indexes else 0 for i in inds[:self.TOP_K]]
            # rel = [self.metric[i, ix] if i in relevant_indexes else 0 for i in inds[:10]]
            ranks.append(rel)

        elif modality == 't2i':
            # This is the same with Hard metric on Threshold = 1
            constant = sum(self.metric[ix, relevant_indexes]) + 1e-20
            rel = [self.metric[ix, i] / constant if i in relevant_indexes else 0 for i in inds[:self.TOP_K]]
            # rel = [self.metric[ix, i] if i in relevant_indexes else 0 for i in inds[:10]]
            ranks.append(rel)

    def softer(self, ix, inds, ranks, modality='i2t', gt_ranks=None):
        if modality == 'i2t':
            # ranks[ix, :] = self.metric[inds[:self.TOP_K]][:, ix]
            ranks[ix, :] = self.metric[inds[:self.TOP_K], ix]
            # For normalization
            gt = list(range(self.TEXT_PER_IMG * ix, self.TEXT_PER_IMG * ix + self.TEXT_PER_IMG, 1))
            inds_metric = np.argsort(self.metric[:, ix])[::-1]
            if not self.include_anns:
                inds_metric = inds_metric[~np.isin(inds_metric, gt)]
                # inds_metric = np.array([i for i in inds_metric if i not in gt])
            # gt_ranks[ix, :] = self.metric[inds_metric[:self.TOP_K]][:, ix]
            gt_ranks[ix, :] = self.metric[inds_metric[:self.TOP_K], ix]

        elif modality == 't2i':
            # Top is 60 times slower!
            # ranks[ix, :] = self.metric[:, inds[:self.TOP_K]][ix, :]
            ranks[ix, :] = self.metric[ix, inds[:self.TOP_K]]
            # For normalization
            inds_metric = np.argsort(self.metric[ix, :])[::-1]
            if not self.include_anns:
                inds_metric = inds_metric[~np.isin(inds_metric, [ix // self.TEXT_PER_IMG])]
                # inds_metric = np.array([i for i in inds_metric if i !=ix//self.TEXT_PER_IMG])
            # gt_ranks[ix, :] = self.metric[:, inds_metric[:self.TOP_K]][ix, :]
            gt_ranks[ix, :] = self.metric[ix, inds_metric[:self.TOP_K]]

    def compute_metrics(self):
        print("\nModel name:{},\n"
              "Dataset: {},\n"
              "Recall Type: {},\n"
              "Metric:{},\n".format(self.model_name, self.dataset, self.recall_type, self.metric_name))
        print("####I2T#####")
        scores_i2t = self.i2t()
        print("####T2I#####")
        scores_t2i = self.t2i()

        return {'i2t': scores_i2t, 't2i': scores_t2i}
