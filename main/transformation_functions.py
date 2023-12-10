import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode_normalize_non_numeric(df: pd.DataFrame, columns_to_process: list) -> pd.DataFrame:
    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()

    le = LabelEncoder()
    scaler = MinMaxScaler()

    for column in columns_to_process:
        if df_encoded[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)

            df_encoded[column] = le.fit_transform(
                df_encoded[column])
            df_encoded[column] = scaler.fit_transform(df_encoded[[column]])

    return df_encoded


def calculate_means(data, column_list):
    means = {}
    for col in column_list:
        sum_values = 0
        count = 0
        for value in data[col]:
            if not pd.isna(value):
                sum_values += value
                count += 1
        if count > 0:
            mean_value = sum_values / count
        else:
            mean_value = None
        means[col] = mean_value
    return means


def custom_mode(x):
    mode_result = x.mode()
    return mode_result.iloc[0] if not mode_result.empty else None
