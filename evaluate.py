from sklearn.metrics import roc_auc_score
import numpy as np

scores=[]
with open("./baselineVsEntropyResult/id_sumExpLogit.txt", "r")as f:
    scores=f.read().split("\n")
    print(scores)

scores.pop()
print('tag1')
with open("./baselineVsEntropyResult/ood_sumExpLogit.txt", "r")as f:
    scores.extend(f.read().split("\n"))
    print(scores)

scores.pop()

float_scores=[]

for s in scores:
    float_scores.append(float(s))

scores=float_scores
print('tag2')
print(scores)


scores2=[]
print('tag3')
with open("./baselineVsEntropyResult/id_entropy.txt", "r")as f:
    scores2=f.read().split("\n")
    print(scores2)

scores2.pop()
print('tag4')
with open("./baselineVsEntropyResult/ood_entropy.txt", "r")as f:
    scores2.extend(f.read().split("\n"))
    print(scores2)

scores2.pop()
print('tag5')
float_scores2=[]

for s in scores2:
    float_scores2.append(float(s))
print('tag6')
scores2=float_scores2
print(scores2)

for i in range(0,20000):
    scores[i]=scores[i]*scores2[i]

label=(np.ones(10000).tolist())+(np.zeros(10000).tolist())
print('tag7')
print(label)

print(roc_auc_score(label,scores))