
import matplotlib.pyplot as plt
import numpy as np
import json

from lib.test_models import test_perfect_model, test_almost_perfect_model, test_gru_model

if __name__=='__main__':

    with open('../data/dataset.json', 'r') as f:
        dataset = json.load(f)

    positives = dataset['positives']
    negatives = dataset['negatives']

    print("Testing perfect...")
    test_perfect_model(positives, negatives)
    print("Testing almost perfect...")
    test_almost_perfect_model(positives, negatives)
    print("Testing GRU model...")
    test_gru_model(positives, negatives, data_fraction=0.5, fp_rate=0.01, pca_embedding_dim=16, lr=0.001, maxlen=50, gru_size=16)


