# 4D-RadarPR
The official repository for "4D RadarPR: Context-Aware 4D Radar Place Recognition in Harsh Scenarios".

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 2.0.1 on Ubuntu 18.04 with CUDA 11.6.1. 

* python 3.8.2
* PyTorch 2.0.1
* pandas 2.0.1
* linear_attention_transformer 0.19.1
* sklearn 

### Training
# Before training:

1. Generate training tuples for Dataset:
    python generating_queries/generate_training_tuple_new_negitives.py --data_path --data_name

2. Generate evaluation tuples:
    python generating_queries/generate_test_sets_radar_new.py --data_path --data_name

# Training:
    eg. Default training parameters on in-house:

python train_radar.py \
    -- dataset_folder \
    -- TRAIN_FILE

    See config.py for all other training parameters.

### evaluation

python evaluate_radar.py \
    --  dataset_folder \
    --  EVAL_QUERY_FILE \
    --  EVAL_DATABASE_FILE


