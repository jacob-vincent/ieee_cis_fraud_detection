import sys

import pandas as pd

from utils import explode_categoricals

input_path = sys.argv[1]
output_path = sys.argv[2]
train_id = pd.read_csv(input_path)
train_id = explode_categoricals(train_id)

train_transaction = pd.read_csv("./data/train_transaction.csv")

train_transaction["P_email_region"] = train_transaction["P_emaildomain"].apply(get_email_region)
train_transaction["P_email_site"] = train_transaction["P_emaildomain"].apply(get_email_site)

train_transaction["R_email_region"] = train_transaction["R_emaildomain"].apply(get_email_region)
train_transaction["R_email_site"] = train_transaction["R_emaildomain"].apply(get_email_site)

train_transaction = explode_categoricals(train_transaction.drop(["P_emaildomain", "R_emaildomain"], axis=1), keep_na=False)

train_df = train_id.merge(right=train_transaction, on="TransactionID", how="inner")

train_df.to_csv(output_path)
