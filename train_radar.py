import torch
import argparse
import sys
from sklearn.neighbors import KDTree

import evaluate_radar as evaluate
import loss.pointnetvlad_loss as PNV_loss

import models.LSR_Place as LSR # normal model for training

import torch.nn as nn
from loading_pointclouds import *
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/', help='results dir [default: results]')
parser.add_argument('--batch_num_queries', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--positives_per_query', type=int, default=2, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18, help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 20]')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Initial learning rate [default: 0.000005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2, help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler', default=0.95)
parser.add_argument('--Data_augmentation', type=bool, default=False, help='use data augmentation')
parser.add_argument('--loss_function', default='quadruplet', choices=['similar_quadruple', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', type=bool, default=True, help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', type=bool, default=True, help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', type=bool, default=False, help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', type=bool, default=False, help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='/path/to/data', help='PointNetVlad Dataset Folder')

FLAGS = parser.parse_args()
cfg.BATCH_NUM_QUERIES = FLAGS.batch_num_queries
cfg.DATA_DIM = 4
cfg.INPUT_DIM = 4
cfg.Data_augmentation = FLAGS.Data_augmentation
cfg.TRAIN_POSITIVES_PER_QUERY = FLAGS.positives_per_query
cfg.TRAIN_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
cfg.MAX_EPOCH = FLAGS.max_epoch
cfg.BASE_LEARNING_RATE = FLAGS.learning_rate
cfg.MOMENTUM = FLAGS.momentum
cfg.OPTIMIZER = FLAGS.optimizer
cfg.DECAY_STEP = FLAGS.decay_step
cfg.DECAY_RATE = FLAGS.decay_rate
cfg.MARGIN1 = FLAGS.margin_1
cfg.MARGIN2 = FLAGS.margin_2
cfg.FEATURE_OUTPUT_DIM = 256

cfg.LOSS_FUNCTION = FLAGS.loss_function
cfg.TRIPLET_USE_BEST_POSITIVES = FLAGS.triplet_use_best_positives
cfg.LOSS_LAZY = FLAGS.loss_not_lazy
cfg.LOSS_IGNORE_ZERO_BATCH = FLAGS.loss_ignore_zero_batch
cfg.RESUME = FLAGS.resume

cfg.RESULTS_FOLDER = FLAGS.results_dir
cfg.DATASET_FOLDER = FLAGS.dataset_folder
data_name = cfg.DATASET_FOLDER.split('/')[-1]

if 'inhouse' in data_name or 'msc' in data_name:
    cfg.NUM_POINTS = 1024
else:
    cfg.NUM_POINTS = 768

cfg.TRAIN_FILE = './generating_queries/radar_split/training_queries_train_inhouse.pickle'

cfg.LOG_DIR = FLAGS.log_dir
if not os.path.exists(cfg.LOG_DIR):
    os.mkdir(cfg.LOG_DIR)
record_file = 'log'
LOG_FOUT = open(os.path.join(cfg.LOG_DIR, data_name + record_file + '.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(os.path.join(BASE_DIR, cfg.TRAIN_FILE))

HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []
TOTAL_ITERATIONS = 0

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(epoch):
    learning_rate = cfg.BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  #


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS

    #设置随机数种子
    # setup_seed(1234)

    if cfg.LOSS_FUNCTION == 'quadruplet':
        loss_function = PNV_loss.quadruplet_loss
    elif cfg.LOSS_FUNCTION == 'similar_quadruple':
        loss_function = PNV_loss.similar_quadruple_loss
    elif cfg.LOSS_FUNCTION == 'Latest_loss':
        loss_function = PNV_loss.Latest_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper

    learning_rate = cfg.BASE_LEARNING_RATE

    train_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'train'))

    model = LSR.LSR_Place(input_dim=cfg.INPUT_DIM, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(parameters, learning_rate, momentum=cfg.MOMENTUM)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if cfg.RESUME:
        resume_filename = cfg.LOG_DIR + "inhouse_250m_last_base.pth"
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        starting_epoch = checkpoint['epoch'] + 1
        # starting_epoch =0

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    scheduler = MultiStepLR(optimizer, milestones=[9, 20], gamma=cfg.DECAY_RATE, last_epoch=-1)

    model = nn.DataParallel(model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    best_recall = 0
    # recall, one_percent_recall = evaluate.evaluate_model(model)

    for epoch in tqdm(range(starting_epoch, cfg.MAX_EPOCH)):
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()
        log_string("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        train_one_epoch(model, optimizer, train_writer, loss_function, epoch)

        if isinstance(model, nn.DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model

        if epoch % 3 == 0 or epoch == cfg.MAX_EPOCH - 1:
            log_string('EVALUATING...')
            cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
            recall, one_percent_recall = evaluate.evaluate_model(model)
            log_string('EVAL RECALL: %s' % str(recall))
            log_string('EVAL one_percent_recall: %s' % str(one_percent_recall))
            train_writer.add_scalar("Val Recall", one_percent_recall, epoch)

            # save BEST model
            if recall[0] >= best_recall:
                best_recall = recall[0]
                save_name = os.path.join(cfg.LOG_DIR, data_name + record_file +'_best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                    save_name)
            log_string('best_top1_recall: %s' % str(best_recall))

        save_name = os.path.join(cfg.LOG_DIR, data_name + record_file +'_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            },
            save_name)
        print("Model Saved As " + save_name)

        scheduler.step()

def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    sampled_neg = 4000
    num_to_take = 10

    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    vaild_train_file_idxs = []
    for vaild_id in train_file_idxs:
        if len(TRAINING_QUERIES[vaild_id]["positives"]) >= cfg.TRAIN_POSITIVES_PER_QUERY:
            vaild_train_file_idxs.append(vaild_id)
    log_string('--number totle files :'+ str(len(train_file_idxs)))
    log_string('--number vaild files :' + str(len(vaild_train_file_idxs)))
    np.random.shuffle(vaild_train_file_idxs)
    loss_sum = 0

    for i in range(len(vaild_train_file_idxs) // cfg.BATCH_NUM_QUERIES):
        batch_keys = vaild_train_file_idxs[i * cfg.BATCH_NUM_QUERIES:(i + 1) * cfg.BATCH_NUM_QUERIES]
        q_tuples = []

        # faulty_tuple = False
        no_other_neg = False

        for j in range(cfg.BATCH_NUM_QUERIES):

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True, data_aug=cfg.Data_augmentation))

            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True, data_aug=cfg.Data_augmentation))

            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(query, negatives, num_to_take)
                hard_negs = list(set().union(HARD_NEGATIVES[batch_keys[j]], hard_negs))
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True, data_aug=True))

            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
                no_other_neg = True
                break

        if no_other_neg:
            continue

        queries, positives, negatives, other_neg = [], [], [], []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)

        if (len(queries.shape) != 4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=cfg.TRIPLET_USE_BEST_POSITIVES, lazy=cfg.LOSS_LAZY, ignore_zero_loss=cfg.LOSS_IGNORE_ZERO_BATCH)

        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if i % 100 ==0 :
            log_string('----' + str(i) + '-----')
            log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += cfg.BATCH_NUM_QUERIES

        # EVALLLL
        if (epoch > 5 and i % 1000 ==0):  ### 决定什么时候+hard
           TRAINING_LATENT_VECTORS = get_latent_vectors(
               model, TRAINING_QUERIES)
           print("Updated cached feature vectors")

        if i == len(vaild_train_file_idxs)//cfg.BATCH_NUM_QUERIES-1:
            log_string('----' + str(i) + '-----')
            log_string('batch loss: %f' % loss)
            epoch_mean_loss = loss_sum / (len(vaild_train_file_idxs)//cfg.BATCH_NUM_QUERIES)
            log_string('epoch loss: %f' % epoch_mean_loss)

def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    queries = np.expand_dims(queries, axis=1)
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        if cfg.INPUT_DIM < 4:
            q = q[:, :, :, :cfg.INPUT_DIM]
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = [TRAINING_LATENT_VECTORS[j] for j in random_negs]

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = cfg.BATCH_NUM_QUERIES * \
        (1 + cfg.TRAIN_POSITIVES_PER_QUERY + cfg.TRAIN_NEGATIVES_PER_QUERY + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = [dict_to_process[index]["query"] for index in file_indices]
        queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        if cfg.INPUT_DIM < 4:
            feed_tensor = feed_tensor[:, :, :, :cfg.INPUT_DIM]
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if len(q_output) != 0:
        q_output = q_output.reshape(-1, q_output.shape[-1])

    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(train_file_idxs)):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            if cfg.INPUT_DIM < 4:
                queries_tensor = queries_tensor[:, :, :, :cfg.INPUT_DIM]
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    return q_output

def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, cfg.NUM_POINTS, cfg.DATA_DIM))
    if cfg.INPUT_DIM < 4:
        feed_tensor = feed_tensor[:,:,:,:cfg.INPUT_DIM]
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(cfg.BATCH_NUM_QUERIES, -1, cfg.FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, cfg.TRAIN_POSITIVES_PER_QUERY, cfg.TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)

    return o1, o2, o3, o4

if __name__ == "__main__":
    train()