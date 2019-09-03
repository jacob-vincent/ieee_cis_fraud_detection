import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def reduce_mem_usage(data_df, sparse=False):
    """Reduce memory usage of Pandas DF.
    From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

    :param data_df: Pandas dataframe
    :type data_df: :class:`pandas.DataFrame`
    :param sparse: Flag to convert data to :class:`scipy.sparse.csr_matrix`
    :type sparse: bool
    :return: Pandas dataframe
    :rtype: :class:`pandas.DataFrame`
    """
    megabyte = 1024**2
    start_mem = data_df.memory_usage().sum()/megabyte
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in data_df.columns:
        col_type = data_df[col].dtype

        if col_type != object:
            c_min = data_df[col].min()
            c_max = data_df[col].max()

            if str(col_type).startswith('int') or str(col_type).startswith('uint'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data_df[col] = data_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data_df[col] = data_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data_df[col] = data_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data_df[col] = data_df[col].astype(np.int64)
            elif str(col_type).startswith('float'):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data_df[col] = data_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data_df[col] = data_df[col].astype(np.float32)
                else:
                    data_df[col] = data_df[col].astype(np.float64)
        else:
            data_df[col] = data_df[col].astype('category')

    if sparse:
        data_df = csr_matrix(data_df)
        end_mem = (data_df.data.nbytes+data_df.indptr.nbytes+data_df.indices.nbytes)/megabyte
    else:
        end_mem = data_df.memory_usage().sum()/megabyte

    percent_change = 100*(start_mem-end_mem)/start_mem
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {percent_change:.1f}%')

    return data_df


def explode_categoricals(data_df, keep_na=True):
    """
    Take a dataframe, find the categorical data stored as strings, and explode those columns into dummy variables.\
    Only variables with less than 10% missing values are kept.

    :param data_df: Input dataframe to examine for categorical data
    :type data_df: :class:`pandas.DataFrame`
    :param keep_na: Flag to keep or ignore NaN values
    :type keep_na: bool
    :return: Input dataframe with categorical data represented as dummy variables
    :rtype: :class:`pandas.DataFrame`
    """

    total_rows = data_df.shape[0]
    # Identify categorical columns not already binarized
    string_columns = []
    for column in data_df.columns:
        if data_df[column].dtype == "O":
            string_columns.append(column)

    # Create dummy variables for categorical columns
    for c in string_columns:
        na_rows = data_df[data_df[c].isna()].shape[0]
        if na_rows/total_rows <= 0.1 and len(data_df[c].value_counts()) > 1 and (1 not in data_df[c].value_counts().values):
            dummy_df = pd.get_dummies(data_df[c], prefix=c, dummy_na=keep_na)
            data_df = pd.concat([data_df, dummy_df], axis=1).drop(c, axis=1)
        else:
            data_df = data_df.drop(c, axis=1)
    return data_df


def get_email_region(text_entry):
    if str(text_entry) == "nan":
        return "nan"
    else:
        return str(text_entry).split(".")[-1]


def get_email_site(text_entry):
    if str(text_entry) == "nan":
        return "nan"
    else:
        return str(text_entry).split(".")[0]


def match_features(input_df, features_array):
    """

    :param input_df:
    :param features_array:
    :return:
    """

    input_columns = list(input_df.columns.values)
    # Add missing features as 0
    for include_feature in features_array:
        if include_feature not in input_columns:
            input_df[include_feature] = 0

    # Find features not present in training data and remove them
    to_exclude = []
    for exclude_feature in input_columns:
        if exclude_feature not in features_array:
            to_exclude.append(exclude_feature)
    input_df = input_df.drop(list(set(to_exclude)), axis=1)

    # Rearrange features into the proper order
    input_df = input_df[features_array]

    if list(input_df.columns.values) == features_array:
        return input_df
    else:
        feature_list = list(input_df.columns.values) + list(features_array)
        for i in feature_list:
            if i not in input_df.columns.values or i not in features_array:
                print(i)
        # assert list(input_df.columns.values) == features_array, "feature mismatch"