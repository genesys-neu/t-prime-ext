import argparse
import sys
import os
sys.path.insert(0, '../')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix as conf_mat
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer, TPrimeDataset_Transformer_overlap
from TPrime_transformer.model_transformer import TransformerModel, TransformerModel_multiclass, TransformerModel_multiclass_transfer
from baseline_models.model_cnn1d import Baseline_CNN1D
from preprocessing.model_rmsnorm import RMSNorm
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix
from conformal_prediction import false_negative_rate, conformal_prediction

# Function to change the shape of obs
# the input is obs with shape (channel, slice)
def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def get_model_name(name):
    name = name.split("/")[-1]
    return '.'.join(name.split(".")[0:-1])

def target_transform(labels):
    return torch.stack([torch.tensor([x, y]) for x, y in zip(labels[0], labels[1])])

def one_hot_encode(index_list):
    num_classes = len(PROTOCOLS)
    encoded_list = []
    index_list = target_transform(index_list)
    for sublist in index_list:
        one_hot = np.zeros(num_classes)
        one_hot[sublist] = 1
        encoded_list.append(one_hot)
    return torch.Tensor(np.array(encoded_list))

def train(model, criterion, optimizer, dataloader, RMSnorm_layer=None):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total_loss = 0
    for batch, (X, y) in tqdm(enumerate(dataloader), desc="Training epochs.."):
        X = X.to(device)
        y = one_hot_encode(y)
        y = y.to(device)
        # Compute prediction error
        if not(RMSnorm_layer is None):
            X = RMSnorm_layer(X)
        pred = model(X.float())
        loss = criterion(pred, y)
        correct += (torch.round(pred) == y).all(dim=1).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    total_loss /= len(dataloader)
    correct /= size
    return correct*100.0, total_loss

def validate(model, criterion, dataloader, RMSnorm_layer=None):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = one_hot_encode(y)
            y = y.to(device)
            if not (RMSnorm_layer is None):
                X = RMSnorm_layer(X)
            pred = model(X.float())
            test_loss += criterion(pred, y).item()
            correct += (torch.round(pred) == y).all(dim=1).type(torch.float).sum().item()
            #y_cpu = y.to('cpu')
            #pred_cpu = pred.to('cpu')
            #conf_matrix += conf_mat(y_cpu, pred_cpu.argmax(1), labels=list(range(nclasses)))
    test_loss /= len(dataloader)
    correct /= size
    return correct*100.0, test_loss#, conf_matrix

def load_params(model, trained):
    # First we will load the params from the pretrained model
    model_dict = model.state_dict()
    pretrained_dict = trained.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    frozen_layers = [k[0] for k in pretrained_dict.items()]
    # Now we will freeze the weights of those params coming from the pretrained model
    for name, param in model.named_parameters():
        if param.requires_grad and name in frozen_layers:
            param.requires_grad = False
            print(name)

    return model

def convert(predictions):
    return [[round(x) for x in sublist] for sublist in predictions]

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def finetune(model, config, trained_model=None):
    # Set seeds
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    # Val-Train split
    dataset = train_val_dataset(ds_train)
    # Create data loaders
    train_dataloader = DataLoader(dataset['train'], batch_size=config['batchSize'], shuffle=True)
    val_dataloader = DataLoader(dataset['val'], batch_size=config['batchSize'], shuffle=True)
    # If we part from an already trained model, load the params into the new one and freeze them
    if config['retrain'] and trained_model is not None:
        model = load_params(model, trained_model)
    if args.use_gpu:
        model.cuda()
    print('Initiating training...')
    # Define loss, optimizer and scheduler for training
    criterion = nn.BCELoss() # Multiclass
    non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    optimizer = torch.optim.Adam(non_frozen_parameters, lr=config['lr']) # CHANGE FOR FINE TUNE
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001)
    train_acc = []
    test_acc = []
    best_acc = 0
    best_cm = 0 # best confusion matrix
    epochs_wo_improvement = 0

    if config['RMSNorm']:
        RMSNorm_l = RMSNorm(model='Transformer')
    else:
        RMSNorm_l = None

    # Training loop
    for epoch in range(config['epochs']):
        acc, loss = train(model, criterion, optimizer, train_dataloader, RMSnorm_layer=RMSNorm_l)
        train_acc.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')
        acc, loss = validate(model, criterion, val_dataloader, RMSnorm_layer=RMSNorm_l)
        test_acc.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (validation)')
        scheduler.step(loss)
        epochs_wo_improvement += 1
        if acc > best_acc:
            best_acc = acc
            epochs_wo_improvement = 0
            # Save model and metrics
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(PATH, MODEL_NAME + '.pt')) #+ '_overlap.pt')) #+ OTA_DATASET + '_' + TEST_FLAG + '_' + RMS_FLAG + NOISE_FLAG + '_ft.pt'))
            #best_cm = conf_matrix
        if epochs_wo_improvement > 12: # early stopping
            print('------------------------------------')
            print('Early termination implemented at epoch:', epoch+1)
            print('------------------------------------')
            break

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./model_cp', help='Path to the trained model or where to save the trained from scratch version \
                        and under which name')
    parser.add_argument("--ds_path", default='', help='Path to the over the air datasets')
    parser.add_argument("--datasets", nargs='+', required=True, help="Dataset names to be used for training or test")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
    parser.add_argument("--transformer", default="lg", choices=["sm", "lg"], help="Size of transformer to use, options available are small and \
                        large. If not defined lg architecture will be used.")
    parser.add_argument("--test_mode", default="random_sampling", choices=["random_sampling", "future"], help="Get test from separate files (future) or \
                        a random sampling of dataset indexes (random_sampling).")
    parser.add_argument("--calib", default=False, action='store_true', help="If present, we calibrate the model with the provided calibration dataset split.")
    parser.add_argument("--retrain", action='store_true', default=False, help="Load the selected model and fine-tune. If this is false the model \
                         will be trained from scratch and the model")
    parser.add_argument("--ota_dataset", default='', help="Flag to add in results name to identify experiment.")
    parser.add_argument("--test", default=False, action='store_true', help="If present, we just test the provided model on OTA data.")
    parser.add_argument("--RMSNorm", default=False, action='store_true', help="If present, we apply RMS normalization on input signals while training and testing")
    parser.add_argument("--back_class", default=False, action='store_true', help="Train/Use model with background or noise class.")
    args, _ = parser.parse_known_args()

    # Config
    MODEL_NAME = get_model_name(args.model_path)
    PATH = '/'.join(args.model_path.split('/')[0:-1])
    CALIB_PATH = './calib_scores/'
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    MIXES = ['b-ax', 'b-g', 'b-n', 'ax-n', 'ax-g', 'n-g']
    CHANNELS = ['None', 'TGn', 'TGax', 'Rayleigh']
    TEST_FLAG = 'rsg' if args.test_mode == 'random_sampling' else 'fut'
    RMS_FLAG = 'RMSn' if args.RMSNorm else ''
    NOISE_FLAG = '_bckg' if args.back_class else ''
    if args.back_class:
        PROTOCOLS.append('noise') 
    OTA_DATASET = args.ota_dataset
    train_config = {
        'batchSize': 122,
        'lr': 0.00002,
        'epochs': 100,
        'retrain': args.retrain,
        'RMSNorm': args.RMSNorm,
        'seed': 1234
    }
    font = {'size': 15}
    plt.rc('font', **font)

    os.makedirs(CALIB_PATH, exist_ok=True)

    datasets = args.datasets
    ds_names = []
    ds_train = []
    ds_calib = []
    ds_test = []
    tr_model = None
    # Load model
    # choose correct version
    if args.retrain:
        global_model = TransformerModel_multiclass_transfer
    else: # no retrain
        global_model = TransformerModel_multiclass
    # choose correct size
    if args.transformer == "sm":
        if args.retrain:
            tr_model = TransformerModel(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2, use_pos=False)
        model = global_model(classes=len(PROTOCOLS), d_model=64*2, seq_len=24, nlayers=2)
        # Load over the air dataset
        for ds in datasets:
            if (os.path.basename(ds) == 'DATASET3_1' or os.path.basename(ds) == 'DATASET3_2' or ds.split('/')[0] == 'DATASET3_3'):
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='calib', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP25')
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='calib', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP50')
            else:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='calib', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    else: # lg
        if args.retrain:
            tr_model = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
        model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2)
        for ds in datasets:
            if (os.path.basename(ds) == 'DATASET3_1' or os.path.basename(ds) == 'DATASET3_2' or ds.split('/')[0] == 'DATASET3_3'):
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='calib', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))                
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP25')
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='calib', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))                
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP50')
            else:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_calib.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='calib', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    
    # concat all loaded datasets
    ds_train = torch.utils.data.ConcatDataset(ds_train)
    ds_calib = torch.utils.data.ConcatDataset(ds_calib)
    if not args.test:
        ds_test = torch.utils.data.ConcatDataset(ds_test)

    print(len(ds_train))
    print(len(ds_calib))
    print(len(ds_test))

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.retrain: # Load pretrained model
        try:
            tr_model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")
    if args.calib and not args.test:
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        calib_sgmd_scores = []
        calib_labels = []
        calib_val_sgmd_scores = []
        calib_val_labels = []
        if train_config['RMSNorm']:
            RMSNorm_l = RMSNorm(model='Transformer')
        else:
            RMSNorm_l = None
        model.to(device)
        model.eval()

        # for ds_ix, ds in enumerate(ds_calib):

        dataset = train_val_dataset(ds_calib)
        # Create data loaders
        calib_dataloader = DataLoader(dataset['train'], batch_size=train_config['batchSize'], shuffle=True)
        calib_val_dataloader = DataLoader(dataset['val'], batch_size=train_config['batchSize'], shuffle=True)

        # calib_dataloader = DataLoader(ds, batch_size=train_config['batchSize'], shuffle=True)
        # print(f'Calibrating model with dataset {ds_names[ds_ix]}')

        # print(ds)
        size = len(calib_dataloader.dataset)
        size_val = len(calib_val_dataloader.dataset)
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(calib_dataloader)):
                # print(X)
                X = X.to(device)
                y = one_hot_encode(y)
                y = y.to(device)
                if not (RMSNorm_l is None):
                    X = RMSNorm_l(X)
                # print(f'X shape: {X.shape}')
                pred = model(X.float())
                # print(f'pred shape: {pred.shape}')
                calib_sgmd_scores.extend(pred.tolist())
                # print(f'y shape: {y.shape}')
                calib_labels.extend(y.tolist())

                if batch % 50 == 0:
                    current = batch * len(X)
                    print(f"[{current:>5d}/{size:>5d}]")

        calib_sgmd_scores = np.array(calib_sgmd_scores)
        calib_labels = np.array(calib_labels)

        print(f'calib sgmd scores{calib_sgmd_scores.shape}')
        print(f'calib labels {calib_labels.shape}')

        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(calib_val_dataloader)):
                # print(X)
                X = X.to(device)
                y = one_hot_encode(y)
                y = y.to(device)
                if not (RMSNorm_l is None):
                    X = RMSNorm_l(X)
                # print(f'X shape: {X.shape}')
                pred = model(X.float())
                # print(f'pred shape: {pred.shape}')
                calib_val_sgmd_scores.extend(pred.tolist())
                # print(f'y shape: {y.shape}')
                calib_val_labels.extend(y.tolist())

                if batch % 50 == 0:
                    current = batch * len(X)
                    print(f"[{current:>5d}/{size_val:>5d}]")

        calib_val_sgmd_scores = np.array(calib_val_sgmd_scores)
        calib_val_labels = np.array(calib_val_labels)

        print(f'calib val sgmd scores{calib_val_sgmd_scores.shape}')
        print(f'calib val labels {calib_val_labels.shape}')


        # Save as numpy array
        np.save(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_sgmd_scores.npy'), calib_sgmd_scores)
        np.save(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_labels.npy'), calib_labels)

        np.save(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_val_sgmd_scores.npy'), calib_val_sgmd_scores)
        np.save(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_val_labels.npy'), calib_val_labels)

    elif args.calib and args.test:
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        if train_config['RMSNorm']:
            RMSNorm_l = RMSNorm(model='Transformer')
        else:
            RMSNorm_l = None
        model.to(device)
        model.eval()

        calib_sgmd_scores = np.load(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_sgmd_scores.npy'))
        calib_labels = np.load(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_labels.npy'))

        calib_val_sgmd_scores = np.load(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_val_sgmd_scores.npy'))
        calib_val_labels = np.load(os.path.join(CALIB_PATH + MODEL_NAME + '_calib_val_labels.npy'))

        print(f'calib sgmd scores{calib_sgmd_scores.shape}')
        print(f'calib labels {calib_labels.shape}')

        print(f'calib val sgmd scores{calib_val_sgmd_scores.shape}')
        print(f'calib val labels {calib_val_labels.shape}')

        # Problem setup
        n=1000 # number of calibration points
        alpha = 0.1 # 1-alpha is the desired false negative rate



        # # 1: get conformal scores. n = calib_Y.shape[0]
        # cal_scores = 1-calib_sgmd_scores[np.arange(n),calib_labels]
        # # 2: get adjusted quantile
        # q_level = np.ceil((n+1)*(1-alpha))/n
        # qhat = np.quantile(cal_scores, q_level, interpolation='higher')
        # prediction_sets = calib_val_sgmd_scores >= (1-qhat) # 3: form prediction sets

        # # Calculate empirical coverage
        # empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]),calib_val_labels].mean()
        # print(f"The empirical coverage is: {empirical_coverage}")
        


        # Run the conformal risk control procedure
        def lamhat_threshold(lam): return false_negative_rate(calib_sgmd_scores>=lam, calib_labels) - ((n+1)/n*alpha - 1/(n+1))
        lamhat = brentq(lamhat_threshold, 0, 1)
        prediction_sets = calib_val_sgmd_scores >= lamhat

        # Calculate empirical FNR
        print(f"The empirical FNR is: {false_negative_rate(prediction_sets, calib_val_labels)} and the threshold value is: {lamhat}")

        for ds_ix, ds in enumerate(ds_test):
            # Calculate performance and save matrix

            # validation loop through test data
            # test_dataloader = DataLoader(ds, batch_size=train_config['batchSize'], shuffle=True)
            test_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
            size = len(test_dataloader.dataset)

            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(device)
                    y = one_hot_encode(y)
                    y = y.to(device)
                    if not (RMSNorm_l is None):
                        X = RMSNorm_l(X)
                    # print(f'X shape: {X.shape}')
                    # print(f'y shape: {y.shape}')
                    pred = model(X.float())
                    # pred = pred.all(dim=0).type(torch.float)
                    # print(f'pred shape: {pred.shape}')
                    prediction_sets = pred > 1-lamhat
                    prediction_sets = prediction_sets.to('cpu').numpy().reshape(-1)
                    print('\n\n')
                    print(f'Total Classes are: {PROTOCOLS}')
                    print(f'Model Prediction is: {pred.cpu()}')
                    print(f'Prediction set is: {prediction_sets}')
                    print(f'True label is: {y.to("cpu").numpy().reshape(-1)}')
                    
                    protocol_list = np.array(PROTOCOLS)
                    print(f'Prediction set is: {list(protocol_list[prediction_sets])}')



    elif args.test and not args.retrain and not args.calib:
        # load model
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        # Use the loaded model to do inference over the OTA dataset
        global_preds = []
        global_trues = []
        global_correct = 0
        global_any_correct = 0
        global_size = 0
        global_noise_preds = []
        noise_predictions = []
        if train_config['RMSNorm']:
                RMSNorm_l = RMSNorm(model='Transformer')
        else:
            RMSNorm_l = None
        model.to(device)
        model.eval()
        for ds_ix, ds in enumerate(ds_test):
            # Calculate performance and save matrix
            preds = []
            trues = []
            # validation loop through test data
            test_dataloader = DataLoader(ds, batch_size=train_config['batchSize'], shuffle=True)
            size = len(test_dataloader.dataset)
            global_size += size
            correct = 0
            any_correct = 0
            noise_preds = []
            with torch.no_grad():
                for X, y in test_dataloader:
                    X = X.to(device)
                    y = one_hot_encode(y)
                    y = y.to(device)
                    if not (RMSNorm_l is None):
                        X = RMSNorm_l(X)
                    pred = model(X.float())
                    correct += (torch.round(pred) == y).all(dim=1).type(torch.float).sum().item()
                    any_correct +=  ((torch.round(pred) == 1) & (y == 1)).any(dim=1).type(torch.float).sum().item()
                    for k in range(len(pred)):
                        if (y[k].to('cpu') == torch.tensor([0, 0, 0, 0, 1])).all():
                            noise_preds.append(torch.round(pred[k]).tolist())
                    y_cpu = y.to('cpu')
                    trues.extend(y_cpu.tolist())
                    pred_cpu = pred.to('cpu')
                    preds.extend(pred_cpu.tolist())
            global_correct += correct
            global_any_correct += any_correct
            global_preds.extend(preds)
            global_trues.extend(trues)
            global_noise_preds.extend(noise_preds)
            correct /= size
            any_correct /= size
            labels = ['ax', 'b', 'n', 'g', 'noise']
            # report accuracy and save confusion matrix
            if ds_names[ds_ix].split(' ')[0] == 'DATASET3_1':
                print(
                    f"\n\nTest Error for dataset {ds_names[ds_ix]}: \n "
                    f"Exact accuracy: {(100 * correct):>0.1f}%, "
                    f"At least one detected: {(100 * any_correct):>0.1f}%, "
                    f"AUC: {roc_auc_score(trues, preds)} \n"
                    f"NOISE classifications: {(100 * sum(np.array(noise_preds))/len(noise_preds))} \n"
                    f"Classification report: {classification_report(trues, convert(preds), labels=np.arange(len(labels)), target_names=labels, zero_division=0)}"
                )
            elif ds_names[ds_ix].split(' ')[0] == 'DATASET3_2':
                print(
                    f"\n\nTest Error for dataset {ds_names[ds_ix]}: \n "
                    f"Exact accuracy: {(100 * correct):>0.1f}%, "
                    f"At least one detected: {(100 * any_correct):>0.1f}%, "
                    f"AUC: {roc_auc_score([row[:4] for row in trues], [row[:4] for row in preds])} \n"
                    #no noise
                    f"Classification report: {classification_report(trues, convert(preds), labels=np.arange(len(labels)), target_names=labels, zero_division=0)}"
                )
            else:
                print(
                    f"\n\nTest Error for dataset {ds_names[ds_ix]}: \n "
                    f"Exact accuracy: {(100 * correct):>0.1f}%, "
                    f"AUC: '-' \n"
                    f"Classification report: {classification_report(trues, convert(preds), labels=np.arange(len(labels)), target_names=labels, zero_division=0)}"
                )
            print('-------------------------------------------')
            print('-------------------------------------------')
        
        # Global confusion matrix for all test datasets if more than one provided
        if len(ds_names) > 1:
            global_correct /= global_size
            global_any_correct /= global_size
            print(
                f"\n\nTest Error for dataset {OTA_DATASET}: \n "
                f"Exact accuracy: {(100 * global_correct):>0.1f}%\n "
                f"At least one detected: {(100 * global_any_correct):>0.1f}%\n "
                f"AUC: {roc_auc_score(global_trues, global_preds)} \n"
                f"NOISE classifications: {(100 * sum(np.array(global_noise_preds))/len(global_noise_preds))} \n"
                f"Classification report: {classification_report(global_trues, convert(global_preds), labels=np.arange(len(labels)), target_names=labels, zero_division=0)}"
                f"Confusion matrices: {multilabel_confusion_matrix(global_trues, convert(global_preds), labels=np.arange(len(labels)))}"
            )
            print('-------------------------------------------')
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config, tr_model)
