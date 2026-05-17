"""
Preprocess dataset for new item (cold start) scenarios.
Generates new_items.npy, cold_items.npy, warm_items.npy and {dataset}_del.inter.
"""
import os
import pandas as pd
import numpy as np
import argparse


def preprocess_new_items(dataset_name, new_item_ratio=0.2):
    """
    Preprocess dataset for new item setting.

    Args:
        dataset_name: Name of dataset (e.g., 'baby')
        new_item_ratio: Ratio of items to be marked as new items (default 0.2)
    """
    df = pd.read_csv(f"{dataset_name}/{dataset_name}.inter", sep='\t')
    n_items = df['itemID'].nunique()

    df1 = df[df['x_label'] == 0]
    item_cnt_df = pd.DataFrame(data={'itemID': np.arange(n_items)})
    item_cnt_df = item_cnt_df.join(df1[['userID', 'itemID']].groupby('itemID').count())
    item_cnt_df.rename(columns={'userID': 'train_cnt'}, inplace=True)
    item_cnt_df.fillna(0, inplace=True)
    item_cnt_df.sort_values('train_cnt', inplace=True)

    # Split items into cold (bottom 70%) and warm (top 30%)
    items_group_cold = np.array(item_cnt_df[:int(n_items * 0.7)]['itemID'].tolist())
    items_group_warm = np.array(item_cnt_df[int(n_items * 0.7):]['itemID'].tolist())

    np.random.seed(1111)
    new_items_cold = np.random.choice(items_group_cold, size=int(new_item_ratio * len(items_group_cold)), replace=False)
    new_items_warm = np.random.choice(items_group_warm, size=int(new_item_ratio * len(items_group_warm)), replace=False)
    new_items = np.concatenate((new_items_cold, new_items_warm))

    os.makedirs(dataset_name, exist_ok=True)
    np.save(f"{dataset_name}/new_items.npy", new_items)
    np.save(f"{dataset_name}/cold_items.npy", items_group_cold)
    np.save(f"{dataset_name}/warm_items.npy", items_group_warm)

    uid_field = 'userID'
    iid_field = 'itemID'
    split = 'x_label'

    cols = [uid_field, iid_field, split]

    df = pd.read_csv(f"{dataset_name}/{dataset_name}.inter", usecols=cols, sep="\t")

    dfs = []
    for i in range(3):
        temp_df = df[df[split] == i].copy()
        dfs.append(temp_df)
    train_u = set(dfs[0][uid_field].values)

    # Loading New Items Index / Removing New items in train/valid
    dfs[2] = pd.concat([dfs[2], dfs[0][dfs[0]['itemID'].isin(new_items)]])
    dfs[0] = dfs[0][~dfs[0]['itemID'].isin(new_items)]

    train_df, valid_df, test_df = dfs

    df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
    df.to_csv(f"{dataset_name}/{dataset_name}_del.inter", sep='\t', index=False)
    print(f"Saved {dataset_name}_del.inter, new_items.npy, cold_items.npy, warm_items.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    args = parser.parse_args()

    preprocess_new_items(args.dataset)