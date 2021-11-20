from scipy.stats import mannwhitneyu
import json
import numpy as np

data_bal = json.load(open("data_balanced_accuracy/liver/liver_test_vector.json", ))
data_BP = json.load(open("data_BP/liver/liver_test_vector.json", ))

#results = mannwhitneyu(data_bal, data_BP, alternative="greater")
data_bal = np.array(data_bal)
data_BP = np.array(data_BP)

print("bal avg =", data_bal.mean())
print("BP avg =", data_BP.mean())
print("bal std =", data_bal.std())
print("BP std =", data_BP.std())
print("bal min =", data_bal.min())
print("BP min =", data_BP.min())
print("bal max =", data_bal.max())
print("BP max =", data_BP.max())

U1, p = mannwhitneyu(data_bal, data_BP, alternative="greater")
print("U1 =",U1)
print("p =", p)