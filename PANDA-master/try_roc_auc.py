from sklearn.metrics import roc_auc_score

a = [0,1,0,0,1,1]
b = [0.01, 3.8, 0.5, 1.2, 0.4, 2.1]
result = roc_auc_score(a,b)
print(result)