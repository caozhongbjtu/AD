import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import get_args
from utils import create_log_file, log_print, write_config
# from model_capsule import MultiScaleAutoencoder
from model import MultiScaleAutoencoder

from all_datasets.AD_dataloader import PSMSegLoader, MSLSegLoader, SMDSegLoader, SWATSegLoader, TSADSegLoader

from metrics.classification_metrics_label import rf, affiliation_f
from metrics.classification_metrics_score import VUS_PR

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
    'GECCO': TSADSegLoader,
    'CICIDS': TSADSegLoader,
    'DLR': TSADSegLoader,
    'NYC': TSADSegLoader,
    'Creditcard': TSADSegLoader,
}

def test_model(model, train_loader, test_loader, device, criterion_rec_reduce, f):
    model.eval()

    attens_energy = []

    with torch.no_grad():
        for data, labels_window in train_loader:
            data = data.to(device)

            fused_rec, _, _, _, _, _ = model(data)

            score = torch.mean(criterion_rec_reduce(fused_rec, data), dim=-2)
            score = score.detach().cpu().numpy()

            attens_energy.append(score)

    train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

    attens_energy = []
    test_labels = []

    with torch.no_grad():
        for data, labels_window in test_loader:
            data = data.to(device)

            fused_rec, _, _, _, _, _ = model(data)

            score = torch.mean(criterion_rec_reduce(fused_rec, data), dim=-2)
            score = score.detach().cpu().numpy()

            attens_energy.append(score)
            test_labels.append(labels_window)

    test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)

    gt = np.concatenate(test_labels, axis=0).reshape(-1).astype(int)
    print("gt.shape:",gt.shape)
    ratio_list = [0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4,4.5,5.0,10,11,12,13,14,15,16,17,18,19,20,25,50]

    VUSROC, VUSPR = VUS_PR(gt, test_energy)


    best_af1 = 0
    best_rf1 = 0
    for ratio in ratio_list:
        threshold = np.percentile(combined_energy, 100 - ratio)

        pred = (test_energy > threshold).astype(int)

        affiliation_f1 = affiliation_f(gt, pred)
        rf1 = rf(gt, pred)

        accuracy = accuracy_score(gt, pred)

        precision, recall, F1, support = precision_recall_fscore_support(gt, pred, average='binary')


        if affiliation_f1 > best_af1:
            best_af1 = affiliation_f1
        if rf1 > best_rf1:
            best_rf1 = rf1
    log_print(f,"A_F1&R_F1&VUSPR&VUSROC:")
    log_print(f,f"&{best_af1:.3f}&{best_rf1:.3f}&{VUSPR:.3f}&{VUSROC:.3f}")
    model.train()

    return best_af1

def train_model(model, train_loader, val_loader, test_loader,
                optimizer, criterion, criterion_rec_reduce,
                epochs, device, model_path, f,alpha):

    best_val_loss = float('inf')
    f1_results = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        start_time = time.time()
        for i, (data, _) in enumerate(train_loader):

            data = data.to(device)

            optimizer.zero_grad()

            t_rec, _, _, _, _, consistency_loss = model(data)

            loss_t = criterion(t_rec, data)

            loss = loss_t + alpha * consistency_loss

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        log_print(f,f"Epoch [{epoch+1}/{epochs}] Time: {epoch_time:.2f}s")
        avg_train_loss = train_loss / len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():
            for data, _ in val_loader:

                data = data.to(device)

                t_rec, attn_loss, fused_rec, z_t, z_f, consistency_loss = model(data)

                loss_t = criterion(t_rec, data)

                loss = loss_t + alpha * consistency_loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        log_print(f, f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    log_print(f,'Training Finish!')
    best_f1 = test_model(model, train_loader, test_loader, device, criterion_rec_reduce, f)

    # f1_results.append(best_f1)
    print(model_path)
    torch.save(model.state_dict(), f'{model_path}.pth')

def main():

    args = get_args()

    print("=" * 40)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 40)

    root_path = "all_datasets/anomaly_detection/" + args.data

    time_ = str(datetime.datetime.now())[5:19]

    f, file_path,model_path = create_log_file(args.log,time_, args.data)

    write_config(f, args)

    log_print(f, f"Dataset: {args.data}")
    log_print(f, "=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA = data_dict[args.data]

    train_dataset = DATA(root_path=root_path, win_size=args.win_size, flag="train")
    val_dataset = DATA(root_path=root_path, win_size=args.win_size, flag="val")
    test_dataset = DATA(root_path=root_path, win_size=args.win_size, flag="tests")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)



    model = MultiScaleAutoencoder(
        seq_len=args.win_size,
        encoding_dim=args.encoding_dim,
        beta=args.beta,
        scales= args.scales
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = nn.MSELoss()
    criterion_rec_reduce = nn.MSELoss(reduction='none')

    try:
        train_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            criterion,
            criterion_rec_reduce,
            args.epochs,
            device,
            model_path,
            f,
            args.alpha
        )

    except KeyboardInterrupt:
        log_print(f, "Interrupted by Ctrl+C")

    finally:
        f.close()

    print("Results have been saved to:", file_path)

if __name__ == '__main__':
    main()