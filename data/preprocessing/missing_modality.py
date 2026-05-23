"""
Preprocess dataset for missing modality scenarios.
Generates missing_items_{ratio}_{modality}.npy with item indices for each modality.
"""
import os
import pandas as pd
import numpy as np
import argparse


def split_arr_(arr, k=4):
    """Split array into k equal parts."""
    res = []
    l = len(arr) / k
    check_ = 0
    for i in range(k):
        from_, to_ = int(i * l), int((i + 1) * l)
        res.append(arr[from_:to_])
        check_ += len(res[i])
    assert check_ == len(arr)
    return res


def preprocess_missing_modality(dataset_name, missing_ratio=2/3, missing_modality='all'):
    """
    Preprocess dataset for missing modality setting.

    Args:
        dataset_name: Name of dataset (e.g., 'baby')
        missing_ratio: Ratio of items to have missing modalities (default 2/3)
        missing_modality: Which modality to make missing ('text', 'image', 'all')
    """
    missing_ratio_name = round(missing_ratio, 3)

    # Determine dataset path - support both direct and data/ subdirectory paths
    if os.path.exists(f"data/{dataset_name}/{dataset_name}.inter"):
        dataset_path = f"data/{dataset_name}"
    elif os.path.exists(f"{dataset_name}/{dataset_name}.inter"):
        dataset_path = dataset_name
    else:
        raise FileNotFoundError(
            f"Dataset file not found. Tried:\n"
            f"  - data/{dataset_name}/{dataset_name}.inter\n"
            f"  - {dataset_name}/{dataset_name}.inter\n"
            f"Please ensure the dataset is in the correct location."
        )

    df = pd.read_csv(f"{dataset_path}/{dataset_name}.inter", sep='\t')
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

    new_item_ratio = 0.2

    np.random.seed(1111)
    new_items_cold = np.random.choice(items_group_cold, size=int(new_item_ratio * len(items_group_cold)), replace=False)
    new_items_warm = np.random.choice(items_group_warm, size=int(new_item_ratio * len(items_group_warm)), replace=False)
    new_items = np.concatenate((new_items_cold, new_items_warm))

    all_items = np.arange(n_items)
    old_items = np.setdiff1d(all_items, new_items)
    old_items_cold = np.setdiff1d(items_group_cold, new_items_cold)
    old_items_warm = np.setdiff1d(items_group_warm, new_items_warm)

    np.random.seed(1225)
    # Missing items selected equally for each group: New Cold, New Warm, Old Cold, Old Warm
    missing_items_nc = np.random.choice(new_items_cold, size=int(missing_ratio * len(new_items_cold)), replace=False)
    missing_items_nw = np.random.choice(new_items_warm, size=int(missing_ratio * len(new_items_warm)), replace=False)
    missing_items_oc = np.random.choice(old_items_cold, size=int(missing_ratio * len(old_items_cold)), replace=False)
    missing_items_ow = np.random.choice(old_items_warm, size=int(missing_ratio * len(old_items_warm)), replace=False)
    missing_items = np.concatenate((missing_items_nc, missing_items_nw, missing_items_oc, missing_items_ow))

    # Distribute missing items based on specified modality type
    missing_items_dict = {}

    if missing_modality in ['text', 't']:
        # All missing items are marked as text-only missing
        missing_items_dict['t'] = missing_items
        missing_items_dict['v'] = np.array([])
        missing_items_dict['all'] = np.array([])
        print(f"[Missing Modality] Mode: TEXT-ONLY")
        print(f"  - Total missing items: {len(missing_items)}")
        print(f"  - Items with missing TEXT: {len(missing_items_dict['t'])}")
        print(f"  - Items with missing IMAGE: {len(missing_items_dict['v'])}")
        print(f"  - Items with missing ALL: {len(missing_items_dict['all'])}")

    elif missing_modality in ['image', 'v', 'visual']:
        # All missing items are marked as image-only missing
        missing_items_dict['t'] = np.array([])
        missing_items_dict['v'] = missing_items
        missing_items_dict['all'] = np.array([])
        print(f"[Missing Modality] Mode: IMAGE-ONLY")
        print(f"  - Total missing items: {len(missing_items)}")
        print(f"  - Items with missing TEXT: {len(missing_items_dict['t'])}")
        print(f"  - Items with missing IMAGE: {len(missing_items_dict['v'])}")
        print(f"  - Items with missing ALL: {len(missing_items_dict['all'])}")

    elif missing_modality == 'all':
        # Original behavior: mixed distribution across t, v, and all
        mnc = split_arr_(missing_items_nc)
        mnw = split_arr_(missing_items_nw)
        moc = split_arr_(missing_items_oc)
        mow = split_arr_(missing_items_ow)

        missing_items_dict['t'] = np.concatenate((mnc[0], mnw[0], moc[0], mow[0]))
        missing_items_dict['v'] = np.concatenate((mnc[1], mnw[1], moc[1], mow[1]))
        missing_items_dict['all'] = np.concatenate((mnc[2], mnw[2], moc[2], mow[2],
                                                    mnc[3], mnw[3], moc[3], mow[3]))
        print(f"[Missing Modality] Mode: ALL (mixed distribution)")
        print(f"  - Total missing items: {len(missing_items)}")
        print(f"  - Items with missing TEXT only: {len(missing_items_dict['t'])}")
        print(f"  - Items with missing IMAGE only: {len(missing_items_dict['v'])}")
        print(f"  - Items with missing ALL modalities: {len(missing_items_dict['all'])}")

    else:
        raise ValueError(f"Invalid missing_modality: {missing_modality}. Must be 'text', 'image', or 'all'")

    # Generate filename with optional modality suffix
    if missing_modality == 'all':
        filename = f"{dataset_path}/missing_items_{missing_ratio_name}"
    else:
        modality_suffix = missing_modality.upper() if missing_modality not in ['t', 'v'] else ('TEXT' if missing_modality == 't' else 'IMAGE')
        filename = f"{dataset_path}/missing_items_{missing_ratio_name}_{modality_suffix}"

    os.makedirs(dataset_path, exist_ok=True)
    np.save(f"{filename}.npy", missing_items_dict, allow_pickle=True)
    print(f"\n[Saved] {filename}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate missing modality datasets')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='dataset name')
    parser.add_argument('--missing_ratio', '-r', type=float, default=0.666, help='missing ratio')
    parser.add_argument('--modality', '-m', type=str, default='all',
                        choices=['text', 't', 'image', 'v', 'visual', 'all'],
                        help='which modality to make missing (text/image/all)')
    args = parser.parse_args()

    preprocess_missing_modality(args.dataset, args.missing_ratio, args.modality)
