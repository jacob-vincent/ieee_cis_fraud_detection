import sys
import pandas as pd
import pickle
from utils import explode_categoricals, get_email_region, get_email_site, match_features

id_path = sys.argv[1]
transaction_path = sys.argv[2]
model_name = sys.argv[3]
header_path = sys.argv[4]
output_path = sys.argv[5]

print("Reading test identity data...")
test_id = pd.read_csv(id_path)
test_id = explode_categoricals(test_id)
print(test_id.shape)

print("Reading test transaction data...")
test_transaction = pd.read_csv(transaction_path)

test_transaction["P_email_region"] = test_transaction["P_emaildomain"].apply(get_email_region)
test_transaction["P_email_site"] = test_transaction["P_emaildomain"].apply(get_email_site)

test_transaction["R_email_region"] = test_transaction["R_emaildomain"].apply(get_email_region)
test_transaction["R_email_site"] = test_transaction["R_emaildomain"].apply(get_email_site)

test_transaction = explode_categoricals(test_transaction.drop(["P_emaildomain", "R_emaildomain"], axis=1))
print(test_transaction.shape)

test_df = test_id.merge(right=test_transaction, on="TransactionID", how="right")
assert test_df.shape[0] == 506691

print("Reading in model...")
with open(f"data/{model_name}", "rb") as file:
    model = pickle.load(file)

print("Reading in model header...")
with open(f"data/{header_path}", "r") as file:
    features = file.read().splitlines()

x_test = match_features(test_df, features)
# y_prob = model.predict_proba(x_test)[:, 1]
#
# submission_df = x_test[["TransactionID"]]
# submission_df["isFraud"] = y_prob
# assert submission_df.shape[0] == 506691, "Submission file does not have the proper number of rows"
# submission_df.to_csv(output_path, index=False)
