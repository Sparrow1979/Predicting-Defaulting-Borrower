import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', False)


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and display missing values in a DataFrame.

    Parameters:
    df (pd_core_frame.DataFrame): The DataFrame for which missing values are to be calculated.

    Returns:
    pd_core_frame.DataFrame: A DataFrame containing the count and percentage of missing values for each column.
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("The dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


def outlier_check(df: pd.DataFrame) -> None:
    """
    Function to check outliers
    :param dataframe: dataframe
    :return: outlier
    """
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].dtype != 'str' and df[col].dtype != 'bool' and df[col].dtype != 'category':
            if col != 'SK_ID_CURR' and col != 'TARGET':
                print(f'Outlier in {col} column')
                sns.boxplot(df[col])
                plt.show()


def boxplotting(df, num_cols, sample_size=100):

    default_group = df[df["TARGET"] == 1]
    normal_group = df[df["TARGET"] == 0]
    default_sample = default_group.sample(n=sample_size, random_state=42)
    normal_sample = normal_group.sample(n=sample_size, random_state=42)

    fig, axes = plt.subplots(nrows=len(num_cols), ncols=2,
                             figsize=(12, 4 * len(num_cols)))

    for i, col in enumerate(num_cols):
        sns.boxplot(data=default_sample, x=col, ax=axes[i, 0])
        sns.boxplot(data=normal_sample, x=col, ax=axes[i, 1])

        axes[i, 0].set_title(f'{col} (Defaulted Loans)')
        axes[i, 1].set_title(f'{col} (Normal Loans)')
        axes[i, 0].set_xlabel('')
        axes[i, 1].set_xlabel('')
        axes[i, 0].set_ylabel('Value')
        axes[i, 1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()


def remove_outliers(df, cols_to_filter):

    df_filtered = df.copy()

    for col in cols_to_filter:
        if col == "TARGET":
            # Skip the "TARGET" column
            continue

        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (
            df_filtered[col] <= upper_bound)]

    return df_filtered


def fill_columns(df, ending, fill_values):
    for column in df.columns:
        if column.endswith(ending):
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(0, inplace=True)
            else:
                fill_value = fill_values.get(column, '')
                df[column].fillna(fill_value, inplace=True)


def grouping(data, group_col):
    df = data.groupby(group_col)['TARGET'].value_counts().unstack()
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={0: 'Normal', 1: 'Defaulted'}, inplace=True)
    return df


def percentaging(data):
    df = data
    df['total'] = df['Normal'] + df['Defaulted']
    df['Normal%'] = round((df['Normal'] / df['total']) * 100, 1)
    df['Defaulted%'] = round((df['Defaulted'] / df['total']) * 100, 1)
    df = df.drop(columns=['Normal', 'Defaulted', 'total'])
    df.rename(columns={'Normal%': 'Normal',
              'Defaulted%': 'Defaulted'}, inplace=True)
    return df


def plot_bar_chart(data, grouping_col, width=10, height=6):
    group = data[grouping_col]
    normal_applicant = data['Normal']
    defaulted_applicants = data['Defaulted']

    bar_width = 0.4

    x = np.arange(len(group))

    title = f'Applicant Behavior By {grouping_col}'

    plt.figure(figsize=(width, height))
    plt.bar(x - bar_width/2, normal_applicant, width=bar_width,
            label='Normal Applicants', align='center')
    plt.bar(x + bar_width/2, defaulted_applicants, width=bar_width,
            label='Defaulted Applicants', align='center')

    plt.xticks(x, group, rotation=90)

    plt.xlabel(grouping_col)
    plt.ylabel('Count')
    plt.title(title)

    plt.legend()

    plt.tight_layout()
    plt.show()


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
