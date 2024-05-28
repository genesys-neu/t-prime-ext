import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
from preprocessing.TPrime_dataset import TPrimeDataset_Transformer, TPrimeDataset_Transformer_overlap
from TPrime_transformer.model_transformer_adapt import TransformerModel_multiclass, TransformerModel_multiclass_LoRA
from sklearn.metrics import classification_report, roc_auc_score

def load_model(model_path, model_class, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    return model

def get_model_name(name):
    name = name.split("/")[-1]
    return '.'.join(name.split(".")[0:-1])

# Function to change the shape of obs
# the input is obs with shape (channel, slice)
def chan2sequence(obs):
    seq = np.empty((obs.size))
    seq[0::2] = obs[0]
    seq[1::2] = obs[1]
    return seq

def test_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predicted = outputs.round()
            total_correct += (predicted == y).all(dim=1).sum().item()
            total_samples += y.size(0)
    accuracy = total_correct / total_samples
    return accuracy

def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def fine_tune_model(model, dataloader, device, lr=0.001):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(5):  # Assuming a small number of epochs for fine-tuning
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser(description="Test and fine-tune models on unseen data.")
    parser.add_argument("--ds_path", default='', help='Path to the over the air datasets')
    parser.add_argument("--datasets", nargs='+', required=True, help="Dataset names to be used for training or test")
    parser.add_argument("--dataset_ratio", default=1.0, type=float, help="Portion of the dataset used for training and validation.")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="Use gpu for fine-tuning and inference")
    parser.add_argument("--transformer", default="lg", choices=["sm", "lg"], help="Size of transformer to use, options available are small and \
                        large. If not defined lg architecture will be used.")
    parser.add_argument("--test_mode", default="random_sampling", choices=["random_sampling", "future"], help="Get test from separate files (future) or \
                        a random sampling of dataset indexes (random_sampling).")
    parser.add_argument("--ota_dataset", default='', help="Flag to add in results name to identify experiment.")
    parser.add_argument("--RMSNorm", default=False, action='store_true', help="If present, we apply RMS normalization on input signals while training and testing")
    parser.add_argument("--back_class", default=False, action='store_true', help="Train/Use model with background or noise class.")
    args, _ = parser.parse_known_args()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--use_lora", action='store_true', help="Use the LoRA architecture for fine-tuning.")
    args = parser.parse_args()

# Config
    MODEL_NAME = get_model_name(args.model_path)
    PATH = '/'.join(args.model_path.split('/')[0:-1])
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets
    ds_names = []
    ds_train = []
    ds_test = []
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
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=24, slice_len=64, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    else: # lg
        model = global_model(classes=len(PROTOCOLS), d_model=128*2, seq_len=64, nlayers=2)
        for ds in datasets:
            if (os.path.basename(ds) == 'DATASET3_1' or os.path.basename(ds) == 'DATASET3_2' or ds.split('/')[0] == 'DATASET3_3'):
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
                                               raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_test.append(TPrimeDataset_Transformer(protocols=PROTOCOLS, ds_path=os.path.join(args.ds_path, ds), ds_type='test', seq_len=64, slice_len=128, slice_overlap_ratio=0, test_ratio=0.2, testing_mode=args.test_mode,
                                              raw_data_ratio=args.dataset_ratio, double_class_label=True, override_gen_map=False, ota=True, apply_wchannel=None, apply_noise=False, transform=chan2sequence))
                ds_names.append(ds)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    if args.use_lora:
        model_class = TransformerModel_multiclass_LoRA
    else:
        model_class = TransformerModel_multiclass
    
    model = load_model(args.model_path, model_class, device)
    initial_accuracy = test_model(model, dataloader, device)
    print(f"Initial Test Accuracy: {initial_accuracy:.2f}")

    if args.use_lora:
        # Freeze original parameters if using LoRA, since we want to learn only the LoRA parameters
        freeze_model_parameters(model)
        # Assume LoRA layers are already integrated and their parameters are set to require_grad=True

    # Fine-tune the model
    fine_tuned_model = fine_tune_model(model, dataloader, device)
    fine_tuned_accuracy = test_model(fine_tuned_model, dataloader, device)
    print(f"Fine-tuned Test Accuracy: {fine_tuned_accuracy:.2f}")

if __name__ == "__main__":
    main()
