import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Modules.model_utils import construct_layers
from collections import OrderedDict
from functools import partial
from constant import *


def MLP(input_size, hidden_sizes):
    layers = []
    layer_idx = 0
    for i, hidden_size in enumerate(hidden_sizes):
        if i == 0:
            rets, layer_idx = construct_layers(
                MLP_CONFIG, input_size, hidden_size, layer_idx
            )
            layers.extend(rets)
        elif i > 0 and i < len(hidden_sizes) - 1:
            rets, layer_idx = construct_layers(
                MLP_CONFIG, hidden_sizes[i - 1], hidden_size, layer_idx
            )
            layers.extend(rets)
        else:
            layers.append(
                (f"layer{layer_idx}", nn.Linear(hidden_sizes[i - 1], hidden_size))
            )
            layer_idx += 1

    layers = nn.Sequential(OrderedDict(layers))

    return layers


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.enc = eval(config)(N_S + N_P, ENC_HIDDEN_SIZES + [N_A + N_O])

    def forward(self, ref_s, ref_p):
        input = torch.cat((ref_s, ref_p), dim=1)
        z = self.enc(input)
        z_a = z[:N_A]
        z_o = z[N_A:]
        return z_a, z_o


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.dec = eval(config)(N_A + N_O, DEC_HIDDEN_SIZES + [N_S + N_P])

    def forward(self, z_a, z_o):
        input = torch.cat((z_a, z_o), dim=1)
        ref_pred = self.dec(input)
        ref_s_pred = ref_pred[:N_S]
        ref_p_pred = ref_pred[N_S]
        return ref_s_pred, ref_p_pred


class Disentangler(nn.Module):
    def __init__(self, config, accent_to_other):
        super(Disentangler, self).__init__()
        if accent_to_other:
            self.disentangler = eval(config)(N_A, A2O_HIDDEN_SIZES + [N_O])
        else:
            self.disentangler = eval(config)(N_O, O2A_HIDDEN_SIZES + [N_A])

    def forward(self, z_a):
        z_o_pred = self.disentangler(z_a)
        return z_o_pred


class Predictor(nn.Module):
    def __init__(self, config):
        super(Predictor, self).__init__()
        predictor, _ = construct_layers(config, N_A, NUM_CLASSES, 0)
        self.predictor = nn.Sequential(OrderedDict(predictor))

    def forward(self, z_a):
        y_a_pred = self.predictor(z_a)
        return y_a_pred


class AccentAdapter(nn.Module):
    def __init__(self):
        super(AccentAdapter, self).__init__()
        self.encoder = Encoder(ENCODER_CONFIG)
        self.decoder = Decoder(DECODER_CONFIG)
        self.predictor = Predictor(ACCENT_PREDICTOR_CONFIG)
        self.a2o_disentangler = Disentangler(
            A2O_DISENTANGLER_CONFIG, accent_to_other=True
        )
        self.o2a_disentangler = Disentangler(
            O2A_DISENTANGLER_CONFIG, accent_to_other=False
        )

    def forward_full(self, ref_s, ref_p):
        z_a, z_o = self.encoder(ref_s, ref_p)
        ref_s_pred, ref_p_pred = self.decoder(z_a, z_o)
        y_a_pred = self.predictor(z_a)
        z_o_pred = self.a2o_disentangler(z_a)
        z_a_pred = self.o2a_disentangler(z_o)
        return z_a, z_o, ref_s_pred, ref_p_pred, y_a_pred, z_o_pred, z_a_pred


# Training loop
def train_step(ref_s, ref_p, y_a, full_networks):
    optimizer_encoder = optim.Adam(full_networks.encoder.parameters(), lr=1e-4)
    optimizer_decoder = optim.Adam(full_networks.decoder.parameters(), lr=1e-4)
    optimizer_classifier = optim.Adam(full_networks.classifier.parameters(), lr=1e-4)
    optimizer_disentangler1 = optim.Adam(
        full_networks.disentangler1.parameters(), lr=1e-4
    )
    optimizer_disentangler2 = optim.Adam(
        full_networks.disentangler2.parameters(), lr=1e-4
    )

    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()

    full_networks.train()

    # Train Encoder, Decoder and Classifier
    # Forward pass
    z_a, z_o, ref_s_pred, ref_p_pred, y_a_pred, z_o_pred, z_a_pred = (
        full_networks.forward_full(ref_s, ref_p)
    )

    # Compute losses
    L1 = criterion_reconstruction(ref_s_pred, ref_s) + criterion_reconstruction(
        ref_p_pred, ref_p
    )
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


# Define Dataset
num_epochs = 100
for epoch in range(num_epochs):
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter()
    ref_s = torch.randn(32, N_S)
    ref_p = torch.randn(32, N_P)
    y_a = torch.randint(0, NUM_CLASSES, (2,))
    for step, batch in enumerate(dataset):  # TODO
        ref_s, ref_p, y_a = batch
        loss_encoder_decoder, L1, L2, L3, L4 = train_step(
            ref_s, ref_p, y_a, full_networks
        )
        # Log the losses to TensorBoard
        writer.add_scalar("Loss/Encoder_Decoder", loss_encoder_decoder.item(), step)
        writer.add_scalar("Loss/L1_Reconstruction", L1.item(), step)
        writer.add_scalar("Loss/L2_Classification", L2.item(), step)
        writer.add_scalar("Loss/L3_Disentangler1", L3.item(), step)
        writer.add_scalar("Loss/L4_Disentangler2", L4.item(), step)

    print(
        f"Epoch {epoch+1}, L1_Reconstruction: {L1}, L2_Classification: {L2}, L3_Disentangler1: {L3}, L4_Disentangler2: {L4}"
    )
    # Save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(full_networks.state_dict(), f"accent_adapter_epoch_{epoch+1}.pth")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":

    full_net = AccentAdapter()
    print(full_net)
