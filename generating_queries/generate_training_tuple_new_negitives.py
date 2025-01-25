# -*-coding:utf-8-*-
import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree

##########################################
# spilt the positive and negative samples of train data
# for quadruplet_loss
# save in the training_queries_long_radar.pickle and training_queries_short_radar.pickle
##########################################

def construct_query_dict(df_centroids, seq_index, queries_len, positive_dist, negative_dist, ues_most_similar):
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=positive_dist)
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=negative_dist)
    queries = {}
    for i in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[i]["data_id"])
        data_file = df_centroids.iloc[i]["data_file"]
        yaw = df_centroids.iloc[i]["yaw"]
        positive_candis = np.setdiff1d(ind_nn[i], [i]).tolist()
        yaw_diff = np.abs(yaw - df_centroids.iloc[positive_candis]["yaw"])
        positives = [c for c in positive_candis if np.min((yaw_diff[c], 360-yaw_diff[c])) < cfgs.yaw_threshold]  # 只留下满足角度的
        most_similar = []
        if ues_most_similar:
            if i-1 in positives:
                most_similar.append(i-1)
            if i+1 in positives:
                most_similar.append(i+1)
            positives = np.setdiff1d(positives, most_similar).tolist()

        non_negative_candis = ind_r[i]
        # non_yaw_diff = np.abs(yaw - df_centroids.iloc[non_negative_candis]["yaw"])
        # non_negatives = [n for n in non_negative_candis if np.min((non_yaw_diff[n], 360-non_yaw_diff[n])) < 120]  # 角度完全不重合的也是负样本
        non_negatives = non_negative_candis
        non_negatives = np.sort(non_negatives)
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), non_negatives).tolist()
        # random.shuffle(non_negatives)
        if seq_index != 0:
            data_id += queries_len
            if ues_most_similar:
                most_similar = [sp + queries_len for sp in most_similar]
            positives = [p + queries_len for p in positives]
            negatives = [n + queries_len for n in negatives]

        if ues_most_similar:
            queries[data_id] = {"query": data_file,"most_similar":most_similar, "positives": positives, "negatives": negatives}
        else:
            queries[data_id] = {"query": data_file, "positives": positives, "negatives": negatives}

    return queries

def split_dataset(base_path, data_name, save_path, positive_dist, negative_dist, use_timestamp_name, ues_most_similar):
    data_path = os.path.join(base_path, data_name)
    groups = sorted(os.listdir(data_path))
    queries_len = 0
    train_seqs = {}
    for group_id, group in enumerate(tqdm(groups)):
        group_dir = os.path.join(data_path, group)
        seqs = [name for name in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, name))]
        df_train = pd.DataFrame(columns=['data_id', 'data_file', 'x', 'y', 'yaw'])
        for seq in tqdm(seqs):
            seq_poses_path = os.path.join(group_dir, seq + '_poses.txt')
            df_locations = pd.read_table(seq_poses_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y',
                                                'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]

            if use_timestamp_name:
                df_locations['data_file'] = data_name + '/' + group + '/' + seq + '/' + df_locations['timestamp'] + '.bin'
            else:
                df_locations['data_id'] = range(0, len(df_locations))
                df_locations['data_file'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
                df_locations['data_file'] = data_name + '/' + group + '/' + seq + '/' + df_locations['data_file'] + '.bin'

            df_train = pd.concat([df_train, df_locations], ignore_index=True)
            df_train['data_id'] = range(0, len(df_train))

        queries = construct_query_dict(df_train, group_id, queries_len, positive_dist, negative_dist, ues_most_similar)
        train_seqs.update(queries)
        queries_len += len(queries)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as handle:
        pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/path/to/data', help='radar datasets path')
    parser.add_argument('--save_folder', type=str, default='./radar_split/', help='the saved path of split file ')
    parser.add_argument('--data_name', type=str, default='train_inhouse', help='train_short or train_long')
    parser.add_argument('--positive_dist', type=float, default=3, help='Positive sample distance threshold, 3m')
    parser.add_argument('--negative_dist', type=float, default=18, help='Negative sample distance threshold, short:10, long:18')
    parser.add_argument('--yaw_threshold', type=float, default=75, help='Yaw angle threshold, 75')
    parser.add_argument('--use_timestamp_name', type=bool, default=False, help='save most similar index for loss')
    parser.add_argument('--ues_most_similar', type=bool, default=False, help='save most similar index for loss')
    cfgs = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_name = 'training_queries_' + cfgs.data_name + '_' + str(cfgs.positive_dist) + 'm_' + str(
        cfgs.yaw_threshold) + '.pickle'
    if cfgs.ues_most_similar:
        save_name = 'training_queries_' + cfgs.data_name + '_' + str(cfgs.positive_dist) + 'm_' + str(
            cfgs.yaw_threshold) + '_similar.pickle'

    save_path = os.path.join(cfgs.save_folder, save_name)
    split_dataset(cfgs.data_path, cfgs.data_name, save_path, cfgs.positive_dist,
                  cfgs.negative_dist, cfgs.use_timestamp_name, cfgs.ues_most_similar)