import numpy as np

def make_adaptive_thresholds(probs, c, g):
    p = np.sort(probs)
    n = len(p)
    widths = [int(np.floor(c ** (np.log(n) / np.log(c))))]

    mini_widths = np.array([c ** (g-k) for k in range(g)])
    mini_widths = n * (mini_widths / mini_widths.sum())
    widths = mini_widths.astype(int)
    if widths.sum() < n:
        widths[0] += n - widths.sum()
    #print(widths)

    thresholds = [0]
    i = 0
    for width in widths:
        i += width
        if i >= n:
            thresholds.append(1 + 1e-10)
            break
        else:
            thresholds.append(p[i] + 1e-10)

    return np.array(thresholds)
    
if __name__ == '__main__':
    preds = np.array([0.9, 0.0, 0.1, 0.2, 0.3, 0.35, 0.7])
    thresholds = np.array([0.0, 0.3, 0.7, 1.0])
    taus = make_adaptive_thresholds(preds, 2, 3)
    np.testing.assert_array_almost_equal(thresholds, taus, 1e-8)

    test_probs = np.array([0.9, 0.0, 0.1, 0.2, 0.3, 0.35])
    thresholds = np.array([0.0, 0.3, 1.0])
    np.testing.assert_array_almost_equal(
            thresholds,
            make_adaptive_thresholds(test_probs, 2, 2), 1e-8)
