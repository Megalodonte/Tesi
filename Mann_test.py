from scipy.stats import mannwhitneyu
import json
import numpy as np
import os
import pandas as pd

datasets_avg_test_accuracy_bal = []
datasets_avg_train_accuracy_bal = []
datasets_test_std_bal = []
datasets_train_std_bal = []
datasets_test_max_bal = []
datasets_train_max_bal = []
datasets_test_min_bal = []
datasets_train_min_bal = []

datasets_avg_test_accuracy_BP = []
datasets_avg_train_accuracy_BP = []
datasets_test_std_BP = []
datasets_train_std_BP = []
datasets_test_max_BP = []
datasets_train_max_BP = []
datasets_test_min_BP = []
datasets_train_min_BP = []
dataset_names = []

U1_test = []
U2_test = []

for name in os.listdir("datasets"):

    data_test_bal = np.array(json.load(open("data_balanced_accuracy/{}/{}_test_vector.json".format(name, name))))
    data_test_BP = np.array(json.load(open("data_BP/{}/{}_test_vector.json".format(name, name))))
    data_train_bal = np.array(json.load(open("data_balanced_accuracy/{}/{}_test_vector.json".format(name, name))))
    data_train_BP = np.array(json.load(open("data_BP/{}/{}_test_vector.json".format(name, name))))

    dataset_names.append(str(name))
    datasets_avg_test_accuracy_bal.append(data_test_bal.mean())
    datasets_avg_train_accuracy_bal.append(data_train_bal.mean())
    datasets_test_std_bal.append(data_test_bal.std())
    datasets_train_std_bal.append(data_train_bal.std())
    datasets_test_max_bal.append(data_test_bal.max())
    datasets_train_max_bal.append(data_train_bal.max())
    datasets_test_min_bal.append(data_test_bal.min())
    datasets_train_min_bal.append(data_train_bal.min())

    datasets_avg_test_accuracy_BP.append(data_test_BP.mean())
    datasets_avg_train_accuracy_BP.append(data_train_BP.mean())
    datasets_test_std_BP.append(data_test_BP.std())
    datasets_train_std_BP.append(data_train_BP.std())
    datasets_test_max_BP.append(data_test_BP.max())
    datasets_train_max_BP.append(data_train_BP.max())
    datasets_test_min_BP.append(data_test_BP.min())
    datasets_train_min_BP.append(data_train_BP.min())

rows_bal_test = [datasets_avg_test_accuracy_bal, datasets_test_std_bal,
                                datasets_test_max_bal, datasets_test_min_bal]

rows_bal_train = [datasets_avg_train_accuracy_bal, datasets_train_std_bal,
                                datasets_train_max_bal, datasets_train_min_bal]

rows_BP_test = [datasets_avg_test_accuracy_BP, datasets_test_std_BP,
                                datasets_test_max_BP, datasets_test_min_BP]

rows_BP_train = [datasets_avg_train_accuracy_BP, datasets_train_std_BP,
                                datasets_train_max_BP, datasets_train_min_BP]

df_bal_test = pd.DataFrame(data=rows_bal_test, index = ["avg", "std", "max", "min"], columns=pd.Index(dataset_names, name="BAL_TEST"))
df_bal_train = pd.DataFrame(data=rows_bal_train, index = ["avg", "std", "max", "min"], columns=pd.Index(dataset_names, name="BAL_TRAIN"))
df_BP_test = pd.DataFrame(data=rows_BP_test, index = ["avg", "std", "max", "min"], columns=pd.Index(dataset_names, name="BP_TEST"))
df_BP_train = pd.DataFrame(data=rows_BP_train, index = ["avg", "std", "max", "min"], columns=pd.Index(dataset_names, name="BP_TRAIN"))

df_bal_test.style
df_bal_train.style
df_BP_test.style
df_BP_train.style

# style = df_bal_test.style
# link = style.render()
# f = codecs.open(link, 'r')
# print(f.read())

#U1, pless = mannwhitneyu(data_test_bal, data_test_BP, alternative="less")
#U1, pless = mannwhitneyu(data_test_bal, data_test_BP, alternative="greater")
#U2 = 225 - U1
#U = min(U1, U2)
