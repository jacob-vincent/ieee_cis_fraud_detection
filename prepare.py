import sys

import pandas as pd

from utils import explode_categoricals, get_email_region, get_email_site

id_path = sys.argv[1]
transaction_path = sys.argv[2]
output_path = sys.argv[3]

train_id = pd.read_csv(id_path)
train_id = explode_categoricals(train_id)
print(train_id.shape)
train_transaction = pd.read_csv(transaction_path)

train_transaction["P_email_region"] = train_transaction["P_emaildomain"].apply(get_email_region)
train_transaction["P_email_site"] = train_transaction["P_emaildomain"].apply(get_email_site)

train_transaction["R_email_region"] = train_transaction["R_emaildomain"].apply(get_email_region)
train_transaction["R_email_site"] = train_transaction["R_emaildomain"].apply(get_email_site)

train_transaction = explode_categoricals(train_transaction.drop(["P_emaildomain", "R_emaildomain"], axis=1), keep_na=False)
print(train_transaction.shape)
train_df = train_id.merge(right=train_transaction, on="TransactionID", how="right")
print(train_df.shape)
train_df.to_csv(output_path, index=False)
