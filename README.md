# 4D-RadarPR
The official repository for "4D RadarPR: Context-Aware 4D Radar Place Recognition in Harsh Scenarios".

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 2.0.1 on Ubuntu 18.04 with CUDA 11.6.1. 

* python 3.8.2
* PyTorch 2.0.1
* pandas 2.0.1
* linear_attention_transformer 0.19.1
* sklearn 

### Dataset
This article uses four datasets:

1. [In-house data (Snail radar)](https://github.com/snail-radar/snail-radar.github.io): Long range radar

2. [Tsinghua data](https://github.com/thucyw/mmWave-Radar-Relocalization/tree/main/dataset): Short range radar

3. [MSC data](https://mscrad4r.github.io): Long range radar

4. [Colorado data](https://arpg.github.io/coloradar/): Short range radar

### Before Training

#### Generate training tuples for Dataset:
```bash
python generating_queries/generate_training_tuple_new_negitives.py --data_path --data_name
```

#### Generate evaluation tuples:
```bash
python generating_queries/generate_test_sets_radar_new.py --data_path --data_name
```

### Training

#### Example: Default training parameters on in-house:
```bash
python train_radar.py \
    -- dataset_folder \
    -- TRAIN_FILE
```
See `config.py` for all other training parameters.

### Evaluation
```bash
python evaluate_radar.py \
    -- dataset_folder \
    -- EVAL_QUERY_FILE \
    -- EVAL_DATABASE_FILE


