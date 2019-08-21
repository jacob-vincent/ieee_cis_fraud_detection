import sys
import pandas as pd
import pickle
from utils import explode_categoricals, get_email_region, get_email_site

id_path = sys.argv[1]
transaction_path = sys.argv[2]
model_name = sys.argv[3]
output_path = sys.argv[4]

test_id = pd.read_csv(id_path)
test_id = explode_categoricals(test_id)

test_transaction = pd.read_csv(transaction_path)

test_transaction["P_email_region"] = test_transaction["P_emaildomain"].apply(get_email_region)
test_transaction["P_email_site"] = test_transaction["P_emaildomain"].apply(get_email_site)

test_transaction["R_email_region"] = test_transaction["R_emaildomain"].apply(get_email_region)
test_transaction["R_email_site"] = test_transaction["R_emaildomain"].apply(get_email_site)

test_transaction = explode_categoricals(test_transaction.drop(["P_emaildomain", "R_emaildomain"], axis=1), keep_na=False)

test_df = test_id.merge(right=test_transaction, on="TransactionID", how="inner")

# test_df.to_csv(output_path)
with open(f"data/{mode_name}", "rb") as file:
    model = pickle.load(file)

x_test = test_df.drop(["TransactionID"], axis=1)
y_prob = model.predict_proba(x_test)[:, 1]

submission_df = x_test[["TransactionID"]]
submission_df["isFraud"] = y_prob
submission_df.to_csv(output_path)
