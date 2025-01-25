import os
import pickle
import numpy as np
import pandas as pd
import argparse
import tqdm
from sklearn.neighbors import KDTree

##########################################
# split query and database data
# save in evaluation_database.pickle / evaluation_query.pickle
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        # pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(output, handle, protocol=1)
    print("Done ", filename)

def construct_query_and_database_sets(base_path, data_name, seqs, positive_dist, yaw_threshold, use_yaw, save_folder, use_timestamp_name):
    database_trees = []
    database_sets = {}
    query_sets = {}

    for seq_id, seq in enumerate(tqdm.tqdm(seqs)):
        seq_path = base_path + data_name + '/' + seq
        tras = [name for name in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, name))]
        tras.sort()
        query = {}
        for tra_id in range(len(tras)):
            pose_path = seq_path + '/' + tras[tra_id] + '_poses.txt'
            df_locations = pd.read_table(pose_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp','r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]

            if use_timestamp_name:
                df_locations['timestamp'] = data_name + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['timestamp'] + '.bin'
                df_locations = df_locations.rename(columns={'timestamp': 'data_file'})
            else:
                df_locations['data_file'] = range(0, len(df_locations))
                df_locations['data_file'] = df_locations['data_file'].apply(lambda x: str(x).zfill(6))
                df_locations['data_file'] = data_name + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['data_file'] + '.bin'

            if tra_id == 0:
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y']])
                database_trees.append(database_tree)
                database = {}
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
                database_sets[seq] = database
            else:
                df_test = df_locations
                for index, row in df_test.iterrows():
                    query[len(query.keys())] = {'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
        query_sets[seq] = query

        for key in range(len(query_sets[seq].keys())):
            coor = np.array([[query_sets[seq][key]["x"], query_sets[seq][key]["y"]]])
            index = database_trees[seq_id].query_radius(coor, r=positive_dist)[0].tolist()
            if use_yaw:
                yaw = query_sets[seq][key]["yaw"]
                yaw_diff = np.abs(yaw - df_database.iloc[index]["yaw"])
                true_index = [c for c in index if np.min((yaw_diff[c], 360-yaw_diff[c])) < cfgs.yaw_threshold]
            else:
                true_index = index
            query_sets[seq][key][seq] = true_index

    os.makedirs(save_folder, exist_ok=True)
    if use_yaw:
        output_to_file(database_sets, save_folder + 'evaluation_database_' + data_name + '.pickle')
        output_to_file(query_sets,
                       save_folder + 'evaluation_query_' + data_name + '_' + str(positive_dist) + 'm_' + str(yaw_threshold) + '.pickle')
    else:
        output_to_file(database_sets, save_folder + 'evaluation_database_' + data_name + '.pickle')
        output_to_file(query_sets, save_folder + 'evaluation_query_' + data_name + '_' + str(positive_dist) + 'm.pickle')

# Building database and query files for evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='path/to/data/',
                        help='radar datasets path')
    parser.add_argument('--save_folder', type=str, default='./radar_split/', help='the saved path of split file ')
    parser.add_argument('--data_name', type=str, default='test_inhouse', help='test data name')
    parser.add_argument('--positive_dist', type=float, default=3,
                        help='Positive sample distance threshold, short:5, long:9')
    parser.add_argument('--yaw_threshold', type=float, default=15, help='Yaw angle threshold, 8 or 25')
    parser.add_argument('--use_yaw', type=bool, default=True, help='If use yaw to determine a positive sample.')
    parser.add_argument('--use_timestamp_name', type=bool, default=False, help='save most similar index for loss')
    cfgs = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if 'test' in cfgs.data_name:
        if "msc" in cfgs.data_name:
            seqs = ['RURAL_A', 'RURAL_B',  'RURAL_D', 'RURAL_E']
        elif "qinghua" in cfgs.data_name:
            seqs = ['seq10', 'seq11'] #'seq3', 'seq4', 'seq6','seq8', 'seq9'
        elif "coloradar" in cfgs.data_name:
            seqs = ['edgar_army_0', 'edgar_army_1', 'edgar_classroom', 'outdoors']
        elif "inhouse" in cfgs.data_name:
            seqs = ['xinghu_narrow', 'xinbu_handheld', 'wenli', 'gongxue' ]
        elif "short" in cfgs.data_name:
            seqs = ['seq8', 'seq9', 'seq3', 'seq4', 'seq6', 'edgar_army', 'edgar_classroom1', 'edgar_classroom2', 'outdoors']
        else:
            seqs = ['RURAL_E', 'xinbu']
    else:
        if "inhouse" in cfgs.data_name:
            seqs = ['xinghu_narrow_1', 'xinbu_handheld_1']
        elif "qinghua" in cfgs.data_name:
            seqs = ['seq1', 'seq2', 'seq5', 'seq7']
        else:
            raise Exception('Loading error!')

    construct_query_and_database_sets(cfgs.data_path, cfgs.data_name, seqs, cfgs.positive_dist, cfgs.yaw_threshold,
                                      cfgs.use_yaw, cfgs.save_folder, cfgs.use_timestamp_name)
