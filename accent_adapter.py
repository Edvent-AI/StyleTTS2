import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
                self.layers.append(nn.BatchNorm1d(hidden_size))
                self.layers.append(nn.ReLU())
            elif i>0 and i<len(hidden_sizes)-1:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
                self.layers.append(nn.BatchNorm1d(hidden_size))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_s, n_p, n_a, n_o, hidden_sizes):
        super(Encoder, self).__init__()
        self.enc = MLP(n_s+n_p, hidden_sizes + [n_a+n_o])
    
    def forward(self, ref_s, ref_p):
        input = torch.cat((ref_s, ref_p), dim=1)
        z = self.enc(input)
        z_a = z[:,:n_a]
        z_o = z[:,n_a:]
        return z_a, z_o

class Decoder(nn.Module):
    def __init__(self, n_s, n_p, n_a, n_o, hidden_sizes):
        super(Decoder, self).__init__()
        self.dec = MLP(n_a+n_o, hidden_sizes + [n_s+n_p])

    
    def forward(self, z_a, z_o):
        input = torch.cat((z_a, z_o), dim=1)
        ref_pred = self.dec(input)
        ref_s_pred = ref_pred[:,:n_s]
        ref_p_pred = ref_pred[:,n_s:]
        return ref_s_pred, ref_p_pred

class Classifier(nn.Module):
    def __init__(self, n_a, n_classes):
        super(Classifier, self).__init__()
        self.classifier =nn.Linear(n_a, n_classes)
    
    def forward(self, z_a):
        y_a_pred = self.classifier(z_a)
        return y_a_pred

class Disentangler(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Disentangler, self).__init__()
        self.disentangler = MLP(input_size, hidden_sizes + [output_size])
    
    def forward(self, z_a):
        z_o_pred = self.disentangler(z_a)
        return z_o_pred

class AccentAdapter(nn.Module):
    def __init__(self, n_s, n_p, n_a, n_o, num_classes, enc_hidden_sizes, dec_hidden_sizes, dis1_hidden_sizes, dis2_hidden_sizes):
        super(AccentAdapter, self).__init__()
        self.encoder = Encoder(n_s, n_p, n_a, n_o, enc_hidden_sizes)
        self.decoder = Decoder(n_s, n_p, n_a, n_o, dec_hidden_sizes)
        self.classifier = Classifier(n_a, num_classes)
        self.disentangler1 = Disentangler(n_a, n_o, dis1_hidden_sizes)
        self.disentangler2 = Disentangler(n_o, n_a, dis2_hidden_sizes)


    def forward_full(self, ref_s, ref_p):
        z_a, z_o = self.encoder(ref_s, ref_p)
        ref_s_pred, ref_p_pred = self.decoder(z_a, z_o)
        y_a_pred = self.classifier(z_a)
        z_o_pred = self.disentangler1(z_a)
        z_a_pred = self.disentangler2(z_o)
        return z_a, z_o, ref_s_pred, ref_p_pred, y_a_pred, z_o_pred, z_a_pred

    def forward(self, ref_s, ref_p): #TODO
        z_a, z_o = self.encoder(ref_s, ref_p)
        ref_s_pred, ref_p_pred = self.decoder(z_a, z_o)
        return z_a, z_o


# Training loop
def train_step(ref_s, ref_p, y_a, full_networks):
    optimizer_encoder = optim.Adam(full_networks.encoder.parameters(), lr=1e-4)
    optimizer_decoder = optim.Adam(full_networks.decoder.parameters(), lr=1e-4)
    optimizer_classifier = optim.Adam(full_networks.classifier.parameters(), lr=1e-4)
    optimizer_disentangler1 = optim.Adam(full_networks.disentangler1.parameters(), lr=1e-4)
    optimizer_disentangler2 = optim.Adam(full_networks.disentangler2.parameters(), lr=1e-4)

    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    
    full_networks.train()

    # Train Encoder, Decoder and Classifier
    # Forward pass
    z_a, z_o, ref_s_pred, ref_p_pred, y_a_pred, z_o_pred, z_a_pred = full_networks.forward_full(ref_s, ref_p)
    # Compute losses
    L1 = criterion_reconstruction(ref_s_pred, ref_s) + criterion_reconstruction(ref_p_pred, ref_p)
    L2 = criterion_classification(y_a_pred, y_a)
    L3 = criterion_reconstruction(z_o_pred.detach(), z_o)
    L4 = criterion_reconstruction(z_a_pred.detach(), z_a)

    # Optimise Encoder, Decoder and Classifier
    loss_encoder_decoder = L1 + L2 - L3 - L4

    loss_encoder_decoder.backward()
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    optimizer_classifier.zero_grad()
    optimizer_encoder.step()
    optimizer_decoder.step()
    optimizer_classifier.step()

    
    # Train Disentangler1 and Disentangler2
    # Forward pass
    z_o_pred = full_networks.disentangler1(z_a.detach())
    z_a_pred = full_networks.disentangler2(z_o.detach())

    L3 = criterion_reconstruction(z_o_pred, z_o.detach())
    L4 = criterion_reconstruction(z_a_pred, z_a.detach())

    # Optimise Disentangler1 and Disentangler2
    optimizer_disentangler1.zero_grad()
    L3.backward()
    optimizer_disentangler1.step()

    optimizer_disentangler2.zero_grad()
    L4.backward()
    optimizer_disentangler2.step()

    
    return loss_encoder_decoder.item(), L1.item(), L2.item(), L3.item(), L4.item()



class CustomDataset(Dataset):
    def __init__(self, ref_s, ref_p, y):
        self.ref_s = torch.tensor(ref_s, dtype=torch.float32)
        self.ref_p = torch.tensor(ref_p, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Assuming y is for classification

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.ref_s[idx], self.ref_p[idx], self.y[idx]


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
    num_epochs = 1000
    batch_size = 32
    ref_s_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/ref_s_all.npy"
    ref_p_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/ref_p_all.npy"
    label_path = "/mnt/iusers01/fatpou01/compsci01/n70579mp/scratch/datasets/speech/vctk/styletts2/emb/labels.npy"
    model_saved_dir = "logs"
    save_every = 20
    log_every = 100
    if not(os.path.exists(model_saved_dir)):
        os.makedirs(model_saved_dir)
    
    
    writer = SummaryWriter()
    ref_s = np.squeeze(np.load(ref_s_path))
    ref_p = np.squeeze(np.load(ref_p_path))
    labels = np.load(label_path)
    print("Num samples", labels.shape[0])

    y = labels[:, 2] # y[i] =  ['p362_312' 'p362' 'American' 'F' '29']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    dataset = CustomDataset(ref_s, ref_p, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    full_networks = AccentAdapter(n_s, n_p, n_a, n_o, num_classes, enc_hidden_sizes, dec_hidden_sizes, dis1_hidden_sizes, dis2_hidden_sizes).to(device)
    global_step = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            ref_s, ref_p, y_a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            loss_encoder_decoder, L1, L2, L3, L4 = train_step(ref_s, ref_p, y_a, full_networks)
            
            # Log the losses to TensorBoard
            if global_step % log_every == 0:
                writer.add_scalar('Loss/Encoder_Decoder', loss_encoder_decoder, global_step)
                writer.add_scalar('Loss/Reconstruction', L1, global_step)
                writer.add_scalar('Loss/Classification', L2, global_step)
                writer.add_scalar('Loss/Disentangler1', L3, global_step)
                writer.add_scalar('Loss/Disentangler2', L4, global_step)
            global_step+=1
        print(f"Epoch {epoch+1}, L1_Reconstruction: {L1}, L2_Classification: {L2}, L3_Disentangler1: {L3}, L4_Disentangler2: {L4}")
        # Save the model every 10 epochs
        if (epoch + 1) % save_every == 0:
            torch.save(full_networks.state_dict(), os.path.join(model_saved_dir, f'accent_adapter_epoch_{epoch+1}.pth'))
            
    writer.close()
