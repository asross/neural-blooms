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

    def check_many(self, items, preds=None):
        if preds is None:
            return [self.check(x) for x in items]
        else:
            return [self.check(x,p) for x,p in zip(items, preds)]

    def check(self, item, p=None):
        if p is None:
            p = self.model.predict(item)
        if p > self.threshold:
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
        #print("Number of false negatives at bloom time", len(false_negatives))
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


class sLBF():
    def __init__(self, model, budget, tau):
        self.model = model
        self.budget = budget
        self.tau = tau
        self.initial_bloom = None
        self.backup_bloom = None

    @property
    def size(self):
        return self.model.size + self.initial_bloom.size + self.backup_bloom.size

    def check_many(self, items, preds=None):
        if preds is None:
            preds = self.model.predicts(items)
        return [self.check(x,p=p) for x,p in zip(items, preds)]

    def check(self, item, p=None):
        if not self.initial_bloom.check(item): return False
        if p is None: p = self.model.predict(item)
        if p >= self.tau:
            return True
        else:
            return self.backup_bloom.check(item)

    def create_bloom_filters(self, positives, pos_preds, neg_preds):
        fn_indices = np.argwhere(pos_preds < self.tau)[:,0]
        fp_indices = np.argwhere(neg_preds >= self.tau)[:,0]
        Fp = len(fp_indices) / len(neg_preds)
        Fn = len(fn_indices) / len(pos_preds)
        alpha = 0.5**np.log(2)
        
        b2 = int(min(self.budget, len(positives) * Fn * np.log(Fp / ((1-Fp)* ((1/Fn)- 1))) / np.log(alpha)))
        b1 = self.budget - b2

        if b1 == 0:
            self.initial_bloom = NullBloomFilter()
        else:
            self.initial_bloom = SizeBasedBloomFilter(len(positives), b1, string_digest)
            for p in positives:
                self.initial_bloom.add(p)

        if b2 == 0:
            print("Probably should never see this")
            self.backup_bloom = NullBloomFilter()
        else:
            self.backup_bloom = SizeBasedBloomFilter(len(fn_indices), b2, string_digest)
            for j in fn_indices:
                self.backup_bloom.add(positives[j])

class AdaLBF():
    def __init__(self, model, budget, g, c, K1):
        self.model = model
        self.budget = budget
        self.g = g
        self.c = c
        self.K1 = K1

    @property
    def size(self):
        return self.model.size + self.bloom_filter.size

    def check_many(self, items, preds=None):
        if preds is None:
            preds = self.model.predicts(items)
        return [self.check(x,p=p) for x,p in zip(items, preds)]

    def check(self, item, p=None):
        if p is None:
            p = self.model.predict(item)
        for j in range(len(self.thresholds)-1):
            if p >= self.thresholds[j] and p < self.thresholds[j+1]:
                return self.bloom_filter.check(item, count=self.hash_counts[j])

    def create_bloom_filter(self, positives, preds):
        self.thresholds = make_adaptive_thresholds(preds, self.c, self.g)
        self.bloom_filter = SizeBasedBloomFilter(len(positives), self.budget, string_digest)
        self.hash_counts = []
        for i in range(len(self.thresholds)-1):
            if i == len(self.thresholds) - 2:
                count = 0
            else:
                count = max(0, self.K1-i)
            self.hash_counts.append(count)
            for j in np.argwhere((preds >= self.thresholds[i])*(preds < self.thresholds[i+1]))[:,0]:
                self.bloom_filter.add(positives[j], count=count)

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

results = []

for fpr in np.logspace(-3,-0.5,11):
    print("**** LBF target FPR = {} ****".format(fpr))

    lbf = LBF(model, fpr)
    lbf.set_threshold(negatives_dev, neg_preds_dev)
    lbf.create_bloom_filter(positives, pos_preds)
    print("LBF size", lbf.size)
    lbf_fpr = np.mean(lbf.check_many(negatives_test, neg_preds_test))
    print("LBF fpr", lbf_fpr)
    results.append(("LBF", lbf.size, lbf_fpr))
    
    bf = SizeBasedBloomFilter(len(positives), lbf.size, string_digest)
    for p in positives:
        bf.add(p)
    print("BF size", bf.size)
    bf_fpr = np.mean([bf.check(x) for x in negatives_test])
    print("BF FPR", bf_fpr)
    results.append(("BF", bf.size, bf_fpr))

    slbf = sLBF(model, lbf.bloom_filter.size, lbf.threshold)
    slbf.create_bloom_filters(positives, pos_preds, neg_preds_dev)
    print("sLBF size", slbf.size)
    slbf_fpr = np.mean(slbf.check_many(negatives_test, neg_preds_test))
    print("sLBF FPR", slbf_fpr)
    results.append(("sLBF", slbf.size, slbf_fpr))

    print("tuning AdaLBF")
    lowest_fpr = float('inf')
    best_params = None
    best_adalbf = None
    for c in [1.01,1.05,1.1,1.5,2]:
        if lowest_fpr == 0: break
        for K1 in np.arange(bf.hash_count-2, bf.hash_count * 2, 2):
            ada_lbf = AdaLBF(model, lbf.bloom_filter.size, 20, c, K1)
            ada_lbf.create_bloom_filter(positives, pos_preds)
            ada_fpr = np.mean(ada_lbf.check_many(negatives_dev[:10000], neg_preds_dev))
            print(c,K1,ada_fpr)
            if ada_fpr < lowest_fpr:
                lowest_fpr = ada_fpr
                best_params = (c, K1) 
                best_adalbf = ada_lbf
            if lowest_fpr == 0: break
    print("Done tuning")
    adalbf = best_adalbf
    print("AdaLBF size", adalbf.size)
    adalbf_fpr = np.mean(adalbf.check_many(negatives_test, neg_preds_test))
    print("AdaLBF FPR", adalbf_fpr)
    results.append(("AdaLBF", adalbf.size, adalbf_fpr))

with open("./results.csv", "w") as f:
    f.write("Method,Size,FPR")
    for m,size,fpr in results:
        f.write("{},{},{}".format(m,size,fpr))
