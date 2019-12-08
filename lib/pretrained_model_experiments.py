import argparse
import json
import math
from test_models import load_gru_model
from BloomFilter import BloomFilter
from utils import *
import mmh3
import os

class LBF():
    def __init__(self, model, fp_rate):
        self.model = model
        self.threshold = None
        self.bloom_filter = None
        self.fp_rate = float(fp_rate)

    def check(self, item):
        if self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter.check(item)

    def set_threshold(self, negatives, preds=None):
        fp_index = math.ceil((len(negatives) * (1 - self.fp_rate/2)))
        if preds is None:
            preds = self.model.predicts(negatives)
        preds.sort()
        self.threshold = preds[fp_index]

    def create_bloom_filter(self, positives, preds=None):
        false_negatives = []
        if preds is None:
            preds = self.model.predicts(positives)
        for i in range(len(positives)):
            if preds[i] <= self.threshold:
                false_negatives.append(positives[i])
        print("Number of false negatives at bloom time", len(false_negatives))
        self.bloom_filter = BloomFilter(
            len(false_negatives),
            self.fp_rate / 2,
            string_digest
        )
        for fn in false_negatives:
            self.bloom_filter.add(fn)
        print("Created bloom filter")

    def size(self):
        return self.bloom_filter.size + self.model.size



print("Loading dataset")
with open('../data/dataset.json', 'r') as f:
    dataset = json.load(f)
positives = dataset['positives']
negatives = dataset['negatives']
negatives_train = negatives[0: int(len(negatives) * .8)]
negatives_dev = negatives[int(len(negatives) * .8): int(len(negatives) * .9)]
negatives_test = negatives[int(len(negatives) * .9): ]

print("Loading model")
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
FLAGS = parser.parse_args()
model = load_gru_model(model_save_path=FLAGS.model)
print("loaded model")

pos_pred_path = FLAGS.model.replace('.h5', '_pos_preds.npy')
neg_dev_pred_path = FLAGS.model.replace('.h5', '_neg_dev_preds.npy')
neg_test_pred_path = FLAGS.model.replace('.h5', '_neg_test_preds.npy')
if not os.path.exists(pos_pred_path):
    np.save(pos_pred_path, model.predicts(positives))
    print("saved positive preds")
if not os.path.exists(neg_dev_pred_path):
    np.save(neg_dev_pred_path, model.predicts(negatives_dev))
    print("saved negative dev preds")
if not os.path.exists(neg_test_pred_path):
    np.save(neg_test_pred_path, model.predicts(negatives_test))
    print("saved negative test preds")
pos_preds = np.load(pos_pred_path)
neg_preds_dev = np.load(neg_dev_pred_path)
neg_preds_test = np.load(neg_test_pred_path)

bf = BloomFilter(len(positives), 0.01, string_digest)

lbf = LBF(model, 0.01)
lbf.set_threshold(negatives_dev, neg_preds_dev)
lbf.create_bloom_filter(positives, pos_preds)

print(bf.size)
print(lbf.size)

import pdb; pdb.set_trace()
