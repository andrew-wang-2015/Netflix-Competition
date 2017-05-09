#rounder test
import numpy as np

preds = np.loadtxt('predictions.txt')

print preds.shape

f = open('predictions_rounded.txt', 'w')

for pr in preds:
    f.write("%.3f\n" % max(min(round(pr, 2),5),1))
