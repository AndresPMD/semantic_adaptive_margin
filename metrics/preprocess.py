import json
import argparse
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data')
    parser.add_argument('--out', default='./data')
    args = parser.parse_args()

    datasets = ['f30k', 'coco']
    splits = ['dev', 'test']

    for dataset in datasets:
        for split in splits:

            # Get the captions
            caps = []
            PATH = os.path.join(args.root, dataset, split+'_caps.txt')
            with open(PATH, 'r') as f:
                for i in f.readlines():
                    caps.append(i.split('\n')[0])

            if split=='dev' and dataset == 'f30k':
                caps = caps[:5000]

            # Get the ids
            ids = []
            PATH = os.path.join(args.root, dataset, split+'_ids.txt')
            with open(PATH, 'r') as f:
                for i in f.readlines():
                    ids.append(i.split('\n')[0])

            caps_json = []
            for ix, cap in enumerate(zip(*[iter(caps)] * 5)):
                caps_json.append({'image_id': ids[ix*5], 'refs': list(cap), 'test': ''})

            OUT_PATH = os.path.join(args.out, dataset+'_'+split+'.json')
            json.dump(caps_json, open(OUT_PATH, 'w'))