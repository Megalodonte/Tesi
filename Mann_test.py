from scipy.stats import mannwhitneyu
import json
import numpy as np

data_bal = json.load(open("data_balanced_accuracy/blood/blood_test_vector.json", ))
data_BP = json.load(open("data_BP/blood/blood_test_vector.json", ))

#results = mannwhitneyu(data_bal, data_BP, alternative="greater")
data_bal = np.array(data_bal)
data_BP = np.array(data_BP)

print("bal avg =", data_bal.mean)
print("BP avg =", data_BP.mean)