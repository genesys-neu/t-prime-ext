import argparse
import sys
import os
sys.path.insert(0, '../')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix as conf_mat
import matplotlib.pyplot as plt
from tqdm import tqdm
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer, TPrimeDataset_Transformer_overlap
from TPrime_transformer.model_transformer import TransformerModel, TransformerModel_multiclass, TransformerModel_multiclass_transfer
from baseline_models.model_cnn1d import Baseline_CNN1D
from preprocessing.model_rmsnorm import RMSNorm
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix

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
    #if type(label) is not list:
    #    return [label, label]
    #return label

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

def validate(model, criterion, dataloader, nclasses, RMSnorm_layer=None):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    #conf_matrix = np.zeros((nclasses, nclasses))
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

def finetune(model, config, trained_model=None):
    # Create data loaders
    train_dataloader = DataLoader(ds_train, batch_size=config['batchSize'], shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=config['batchSize'], shuffle=True)
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, verbose=True)
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
        acc, loss = validate(model, criterion, test_dataloader, config['nClasses'], RMSnorm_layer=RMSNorm_l)
        test_acc.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f} (test)')
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
    # best_cm = best_cm.astype('float')
    # for r in range(best_cm.shape[0]):  # for each row in the confusion matrix
    #     sum_row = np.sum(best_cm[r, :])
    #     best_cm[r, :] = best_cm[r, :] / sum_row  * 100.0 # compute in percentage
    # print('------------------- Best confusion matrix (%) -------------------')
    # print(np.around(best_cm, decimals=2))
    # prot_display = ['ax', 'b', 'n', 'g'] #PROTOCOLS
    # if len(PROTOCOLS) > 4: # We need to add noise class
    #     prot_display.append('noise')
    # #prot_display[1] = '802_11b'
    # disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=prot_display)
    # disp.plot(cmap="Blues", values_format='.2f')
    # disp.ax_.get_images()[0].set_clim(0, 100)
    # plt.title(f'Conf. Matrix (%): Total Acc. {(best_acc):>0.1f}%')
    # plt.savefig(f"Results_finetune_{MODEL_NAME}_ft.{OTA_DATASET}.{TEST_FLAG}.{RMS_FLAG}{NOISE_FLAG}.pdf")
    # plt.clf()
    # print('-----------------------------------------------------------------')
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
    PROTOCOLS = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
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
        'nClasses': 7, #len(PROTOCOLS),
        'retrain': args.retrain,
        'RMSNorm': args.RMSNorm
    }
    font = {'size': 15}
    plt.rc('font', **font)

    datasets = args.datasets
    ds_names = []
    ds_train = []
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
            if (os.path.basename(ds) == 'DATASET3_1' or os.path.basename(ds) == 'DATASET3_2'):
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP25')
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP50')
            else:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    else: # lg
        if args.retrain:
            tr_model = TransformerModel(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2, use_pos=False)
        model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2)
        for ds in datasets:
            if (os.path.basename(ds) == 'DATASET3_1' or os.path.basename(ds) == 'DATASET3_2'):
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP25'), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP25')
                ds_train.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer_overlap(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds, 'OVERLAP50'), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                                raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds + ' OVERLAP50')
            else:
                ds_train.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='train', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                               raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    
    # concat all loaded datasets
    ds_train = torch.utils.data.ConcatDataset(ds_train)
    if not args.test:
        ds_test = torch.utils.data.ConcatDataset(ds_test)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.retrain: # Load pretrained model
        try:
            tr_model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        except:
            raise Exception("The model you provided does not correspond with the selected architecture. Please revise and try again.")

    if args.test and not args.retrain:
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
        MIXES = ['b-ax', 'b-g', 'b-n', 'ax-n', 'ax-g', 'n-g']
        MIXES_LABS = [[1,1,0,0,0],
                      [0,1,0,1,0],
                      [0,1,1,0,0],
                      [1,0,1,0,0],
                      [1,0,0,1,0],
                      [0,0,1,1,0]]
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
            print(f"\n\nTest Error for dataset {ds_names[ds_ix]}: \n ")
            for mix_index, mix in enumerate(MIXES):
                # Get the positions where that overlap config. should have been identified
                positions = [index for index, sublist in enumerate(trues) if sublist == MIXES_LABS[mix_index]]
                # Get the predictions and convert them to one hot encoding
                mix_preds = convert([preds[index] for index in positions])
                # Add up how many times you predicted well the ax and divide by class size
                ax_acc = sum(sublist[0] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the b and divide by class size
                b_acc = sum(sublist[1] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the n and divide by class size
                n_acc = sum(sublist[2] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the g and divide by class size
                g_acc = sum(sublist[3] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the noise and divide by class size
                noise_acc = sum(sublist[4] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted both well and divide by class size
                exact_acc = sum(1 for sublist in mix_preds if sublist == MIXES_LABS[mix_index])/len(mix_preds)
                # Count how many times we at least have detected one of the protocols and when we detect any of both but not any more classes
                single_acc_restrictive = 0
                single_acc = 0
                for sublist in mix_preds:
                    # Check if the sublist has at least one '1' in the same position as the target list
                    if any(sublist[i] == 1 and MIXES_LABS[mix_index][i] == 1 for i in range(5)):
                        single_acc += 1
                        # Check if all other positions are '0'
                        if not any(sublist[i] == 1 and MIXES_LABS[mix_index][i] == 0 for i in range(5)):
                            single_acc_restrictive += 1
                # Get accuracy
                single_acc_restrictive /= len(mix_preds)
                single_acc /= len(mix_preds)
                print(
                    f"MIX: {mix}\n"
                    f"Exact accuracy: {(100 * exact_acc):>0.1f}%, "
                    f"Single restrictive accuracy: {(100 * single_acc_restrictive):>0.1f}%, "
                    f"Single accuracy: {(100 * single_acc):>0.1f}%\n"
                    f"ax detected in: {(100 * ax_acc):>0.1f}% of mix instances, \n"
                    f"b detected in: {(100 * b_acc):>0.1f}% of mix instances, \n"
                    f"n detected in: {(100 * n_acc):>0.1f}% of mix instances, \n"
                    f"g detected in: {(100 * g_acc):>0.1f}% of mix instances, \n"
                    f"noise detected in: {(100 * noise_acc):>0.1f}% of mix instances, \n"
                )
            print('-------------------------------------------')
            print('-------------------------------------------')
        
        # Global confusion matrix for all test datasets if more than one provided
        if len(ds_names) > 1:
            print(f"\n\nTest Error for GLOBAL dataset: \n ")
            for mix_index, mix in enumerate(MIXES):
                # Get the positions where that overlap config. should have been identified
                positions = [index for index, sublist in enumerate(global_trues) if sublist == MIXES_LABS[mix_index]]
                # Get the predictions and convert them to one hot encoding
                mix_preds = convert([global_preds[index] for index in positions])
                # Add up how many times you predicted well the ax and divide by class size
                ax_acc = sum(sublist[0] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the b and divide by class size
                b_acc = sum(sublist[1] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the n and divide by class size
                n_acc = sum(sublist[2] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the g and divide by class size
                g_acc = sum(sublist[3] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted well the noise and divide by class size
                noise_acc = sum(sublist[4] for sublist in mix_preds)/len(mix_preds)
                # Add up how many times you predicted both well and divide by class size
                exact_acc = sum(1 for sublist in mix_preds if sublist == MIXES_LABS[mix_index])/len(mix_preds)
                # Count how many times we at least have detected one of the protocols or both but not any more classes
                single_acc_restrictive = 0
                single_acc = 0
                for sublist in mix_preds:
                    # Check if the sublist has at least one '1' in the same position as the target list
                    if any(sublist[i] == 1 and MIXES_LABS[mix_index][i] == 1 for i in range(5)):
                        single_acc += 1
                        # Check if all other positions are '0'
                        if not any(sublist[i] == 1 and MIXES_LABS[mix_index][i] == 0 for i in range(5)):
                            single_acc_restrictive += 1
                # Get accuracy
                single_acc_restrictive /= len(mix_preds)
                single_acc /= len(mix_preds)
                print(
                    f"MIX: {mix}\n"
                    f"Exact accuracy: {(100 * exact_acc):>0.1f}%, "
                    f"Single restrictive accuracy: {(100 * single_acc_restrictive):>0.1f}%, "
                    f"Single accuracy: {(100 * single_acc):>0.1f}%\n"
                    f"ax detected in: {(100 * ax_acc):>0.1f}% of mix instances, \n"
                    f"b detected in: {(100 * b_acc):>0.1f}% of mix instances, \n"
                    f"n detected in: {(100 * n_acc):>0.1f}% of mix instances, \n"
                    f"g detected in: {(100 * g_acc):>0.1f}% of mix instances, \n"
                    f"noise detected in: {(100 * noise_acc):>0.1f}% of mix instances, \n"
                )
            print('-------------------------------------------')
            print('-------------------------------------------')
            print('-------------------------------------------')
    else:
        # Fine-tune the provided model with the new data
        finetune(model, train_config, tr_model)
