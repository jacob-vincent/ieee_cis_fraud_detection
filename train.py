import pickle
import sys

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

input_path = sys.argv[1]
model_file_name = sys.argv[2]
header_path = sys.argv[3]
metrics_path = sys.argv[4]

print("Reading training data...")
train_df = pd.read_csv(input_path)

train_df, val_df = train_test_split(train_df)
print(train_df.shape)
print(val_df.shape)

x_train, y_train = train_df.drop(["TransactionID","isFraud"], axis=1), train_df[["TransactionID", "isFraud"]]
x_val, y_val = val_df.drop(["TransactionID","isFraud"], axis=1), val_df[["TransactionID", "isFraud"]]

# x_train = reduce_mem_usage(x_train, sparse=True)

# x_val = reduce_mem_usage(x_val, sparse=True)
print("Training model...")
model = lgb.LGBMClassifier(boosting_type="goss", n_estimators=500, min_data_in_leaf=3, objective="binary", silent=False, random_state=42)
trained_model = model.fit(x_train, y_train["isFraud"].values)

print("Saving model...")
with open(f"./data/{model_file_name}", "wb") as file:
    pickle.dump(trained_model, file)

print("Saving model header file...")
with open(f"./data/{header_path}", "w") as file:
    for i in x_train.columns:
        file.write(i)
        file.write("\n")


y_prob = trained_model.predict_proba(x_val)[:, 1]
roc_auc = roc_auc_score(y_val["isFraud"].values, y_prob)
pr_auc = average_precision_score(y_val["isFraud"].values, y_prob)

with open(f"{metrics_path}/auc.metric", "w") as file:
    file.write(f"{roc_auc: .5f}")

with open(f"{metrics_path}/pr_auc.metric", "w") as file:
    file.write(f"{pr_auc: .5f}")

print(f"ROC-AUC: {roc_auc}")
print(f"PR-AUC: {pr_auc}")
