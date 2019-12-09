import argparse
import json
import math
from test_models import load_gru_model
from BloomFilter import BloomFilter, SizeBasedBloomFilter, NullBloomFilter
from utils import *
import mmh3
import os
from adaptive_thresholds import *

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

    @property
    def size(self):
        return self.bloom_filter.size + self.model.size

class DadaLBF():
    def __init__(self, model, budget, g, c):
        self.model = model
        self.budget = budget
        self.g = g
        self.c = c

    @property
    def size(self):
        return self.model.size + sum([bf.size for bf in self.bloom_filters])

    def check(self, item):
        p = self.model.predict(item)
        for j in range(len(self.thresholds)):
            if p >= self.thresholds[j] and p < self.thresholds[j+1]:
                return self.bloom_filters[j].check(item)

    def create_bloom_filters(self, positives, preds):
        taus = make_adaptive_thresholds(preds, self.c, self.g)
        counts = np.array([
            np.sum((preds >= taus[j])*(preds < taus[j+1])) for j in range(len(taus)-1)
        ])
        sizes = [self.g * len(preds)]
        for j in range(1, len(counts)-1):
            size = counts[j] * (sizes[0]/counts[0] + j*(np.log(self.c)/np.log(0.6185)))
            assert(size >= 0)
            sizes.append(size)
        sizes.append(0)

        sizes = np.array(sizes)
        sizes = sizes / sizes.sum()
        sizes = (self.budget * sizes).astype(int)
        if sizes.sum() < self.budget:
            sizes[0] += self.budget - sizes.sum()
        assert(abs(sizes.sum() - self.budget) < 0.1)

        self.thresholds = taus
        self.bloom_filters = []
        for i in range(len(counts)):
            if sizes[i] == 0:
                bf = NullBloomFilter()
            else:
                bf = SizeBasedBloomFilter(counts[i], sizes[i], string_digest)
                for j in np.argwhere((preds >= taus[i])*(preds < taus[i+1]))[:,0]:
                    bf.add(positives[j])
            self.bloom_filters.append(bf)

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


lbf = LBF(model, 0.01)
lbf.set_threshold(negatives_dev, neg_preds_dev)
lbf.create_bloom_filter(positives, pos_preds)

bf = SizeBasedBloomFilter(len(positives), lbf.size, string_digest)

print(bf.size)
print(lbf.size)

dada_lbf = DadaLBF(model, lbf.bloom_filter.size, 30, 3)
dada_lbf.create_bloom_filters(positives, pos_preds)
print(dada_lbf.size)

bf_fpr = np.mean([bf.check(x) for x in negatives_test[:1000]])
print("BF FPR", bf_fpr)

lbf_fpr = np.mean([lbf.check(x) for x in negatives_test[:1000]])
print("LBF FPR", lbf_fpr)

dada_fpr = np.mean([dada_lbf.check(x) for x in negatives_test[:1000]])
print("Dada FPR", dada_fpr)

import pdb; pdb.set_trace()
