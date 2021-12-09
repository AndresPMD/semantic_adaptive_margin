import argparse
import json
import os
import numpy as np
import pandas as pd

from metric import Metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/data2fast/users/amafla/Precomp_features/data/', help='ground truth data path')

    parser.add_argument('--metric_path', type=str, default='./out', help='the path that has metrics and model output')

    parser.add_argument('--dataset', type=str, default='coco', help='which dataset to use, options are: coco, f30k')
    parser.add_argument('--split', type=str, default='test',
                        help='Choose to evaluate on coco 1k test set or 5k test set. (test | testall)')

    parser.add_argument('--metric_name', type=str, default='spice',
                        help='which image captioning metric to use, options are: cider, spice')

    parser.add_argument('--recall_type', type=str, default='recall', help='Options are recall and vse_recall')

    parser.add_argument('--score', default=['hard', 'soft', 'softer'], nargs="+",
                        help='which scoring method to use, options are: hard, soft, softer')

    parser.add_argument('--model_name', type=str, default='CVSE_cider',
                        help='which model to use, options are: VSEPP, SCAN, VSRN, CVSE')

    parser.add_argument('--threshold', type=int, default=1,
                        help='Threshold of number of relevant samples to compute metrics, options are: 1,2,3')
    parser.add_argument('--recall_thresholds', default=[1, 5, 10, 20, 30], nargs="+", help='K values in Recall_at_K')
    parser.add_argument('--include_anns', action='store_true',
                        help='Include human annotations to define relevant items')

    args = parser.parse_args()

    if args.metric_name == 'spice':
        metric = pd.read_csv(os.path.join(args.metric_path, args.dataset + '_' + args.metric_name + '.csv'), sep=',',
                             header=None)
        metric = metric.to_numpy()
        if args.dataset == 'coco': metric = metric[:, :5000]
        if args.dataset == 'f30k': metric = metric[:, :1000]

    elif args.metric_name == 'cider':
        metric = np.load(os.path.join(args.metric_path, args.dataset + '_cider.npy'))

    if args.split == 'testall' and args.dataset == 'coco':
        metric = metric[:, :5000]
    elif args.split == 'test' and args.dataset == 'coco':
        metric = metric[:, :1000]

    filename = os.path.join(args.metric_path, 'sims_' + args.model_name + '_' + args.dataset + '_precomp.json')
    sims = json.load(open(filename, 'r'))

    if len(sims) == 1000 and args.dataset == 'coco' and args.split == 'testall':
        raise ValueError('You cant have coco 1k and testall option together')
    if len(sims) == 5000 and args.dataset == 'coco' and args.split == 'test':
        raise ValueError('You cant have coco 5k and test option together')

    M = Metric(metric, sims, recall_type=args.recall_type, score=args.score, metric_name=args.metric_name,
               recall_thresholds=args.recall_thresholds, threshold=args.threshold, dataset=args.dataset,
               include_anns=args.include_anns, model_name=args.model_name)

    print("\n ... LOADING DATA ...\n")
    scores = M.compute_metrics()
