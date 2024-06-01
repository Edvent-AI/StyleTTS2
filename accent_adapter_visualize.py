import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
from accent_adapter_wo_adv import *
from constants import *
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE


def create_df_from_numpy(emb, accent_list, spk_id_list, sex_list):
    feat_cols = [ 'tsne_dim_'+str(i) for i in range(emb.shape[1]) ]
    df = pd.DataFrame(emb, columns=feat_cols)
    df['speaker id'] = spk_id_list
    df['accent'] = accent_list
    df['sex'] = sex_list
    return df

def plot(df, emb_type="emb_style", label_type="accent"):
    plt.figure(figsize=(16,10))
    title = {'z_a': 'TSNE plot of z_a',
             'z_o': 'TSNE plot of z_o'}
    sns.scatterplot(
        x="tsne_dim_0", y="tsne_dim_1",
        hue=label_type,
        palette=sns.color_palette("hls", 13),
        data=df,
        legend="full",
        alpha=0.3
    ).set(title=title[emb_type])
    saved_path = os.path.join(saved_dir, emb_type + "_" + label_type + ".png")
    plt.savefig(saved_path)



if __name__ == "__main__":
    # args: TODO
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_s, n_p = 128, 128
    n_a = 16
    n_o = 48
    num_classes = 2
    enc_hidden_sizes = [128,96,64]
    dec_hidden_sizes = [64,96,128]
    dis1_hidden_sizes = [32,32]
    dis2_hidden_sizes = [32,32]
    batch_size = 1024
    ref_s_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/ref_s_all.npy"
    ref_p_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/ref_p_all.npy"
    label_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/labels.npy"
    ckpt_path = "logs/without_adv_old/accent_adapter_epoch_220.pth"
    saved_dir = "./figures/"
    if not(os.path.exists(saved_dir)):
        os.makedirs(saved_dir)
    
    
    ref_s = np.squeeze(np.load(ref_s_path))
    ref_p = np.squeeze(np.load(ref_p_path))
    labels = np.load(label_path)
    y = labels[:, 2] # y[i] =  ['p362_312' 'p362' 'American' 'F' '29']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    spk_id = labels[:, 1]
    sex = labels[:,3]

    # Create a dataframe for easy manipulation
    data = pd.DataFrame({'ref_s': list(ref_s), 'ref_p': list(ref_p), 'y': list(y), 'speaker_id': list(spk_id), "sex":list(sex)})

    # Split the data based on speaker_ids
    all_spk = train_spk + test_spk
    train_data = data[data['speaker_id'].isin(train_spk)]
    test_data = data[data['speaker_id'].isin(test_spk)]
    all_data = data[data['speaker_id'].isin(all_spk)]
    print("Num samples", len(test_data['ref_s']))
    print("Num Test Speakers", len(test_spk))
    # Create datasets
    train_dataset = SpeakerDataset(train_data)
    test_dataset = SpeakerDataset(test_data)
    all_dataset = SpeakerDataset(all_data)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

    full_networks = AccentAdapter(n_s, n_p, n_a, n_o, num_classes, enc_hidden_sizes, dec_hidden_sizes, dis1_hidden_sizes, dis2_hidden_sizes)
    full_networks.load_state_dict(torch.load(ckpt_path))
    full_networks.to(device)
    full_networks.eval()
    # prepare to count predictions for each class
    predictions = []
    labels = []
    z_a_list = []
    z_o_list = []
    accent_list = []
    spk_id_list = []
    sex_list = []
    with torch.no_grad():
        for step, batch in enumerate(all_loader):
            ref_s, ref_p, y_a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            spk_id, sex = batch[3], batch[4]
            z_a, z_o, ref_s_pred, ref_p_pred, y_a_pred = full_networks.forward_full(ref_s, ref_p)
            _, y_a_pred = torch.max(y_a_pred, 1)
            

            y_a_pred = y_a_pred.detach().cpu().numpy().astype(int)
            y_a = y_a.detach().cpu().numpy().astype(int)
            predictions += list(y_a_pred)
            labels += list(y_a)

            # For Visualization
            z_a_list += list(z_a.detach().cpu().numpy())
            z_o_list += list(z_o.detach().cpu().numpy())
            accent_list += list(y_a)
            spk_id_list += spk_id
            sex_list += sex

    z_a_all = np.array(z_a_list)
    z_o_all = np.array(z_o_list)
    # Check
    # print(z_a_all.shape)
    # print(z_o_all.shape)
    # print(len(accent_list))
    # print(len(spk_id_list))
    # print(len(sex_list))
    # print(accent_list[0])
    # print(spk_id_list[0])
    # print(sex_list[0])
    # Accuracy  
    print("Accent Classification Accuracy:", accuracy_score(labels, predictions))
    

    # # # Create TSNE

    z_a_tsne = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=40).fit_transform(z_a_all)
    z_o_tsne = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=40).fit_transform(z_o_all)



    # # # Visualize TSNE
    df_z_a = create_df_from_numpy(z_a_tsne, accent_list, spk_id_list, sex_list)
    df_z_o = create_df_from_numpy(z_o_tsne, accent_list, spk_id_list, sex_list)

    plot(df_z_a, emb_type="z_a", label_type="accent")
    plot(df_z_a, emb_type="z_a", label_type="sex")
    plot(df_z_a, emb_type="z_a", label_type="speaker id")

    plot(df_z_o, emb_type="z_o", label_type="accent")
    plot(df_z_o, emb_type="z_o", label_type="sex")
    plot(df_z_o, emb_type="z_o", label_type="speaker id")
