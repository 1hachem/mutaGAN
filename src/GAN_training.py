import torch
import torch.nn as nn
from tqdm import tqdm

from src.encode import to_ix


def GAN_train(seq2seq, discriminator_encoder, discriminator, optimizer_generator, optimizer_discriminator, parent_data_loader, child_data_loader, num_epochs, device, not_child_data_loader=None):
    for epoch in range(num_epochs):
        progress_bar = tqdm(zip(parent_data_loader, child_data_loader), total=len(parent_data_loader))
        for parent, child in progress_bar:

            parent = parent.to(device)
            child = child.to(device)

            generated_child = seq2seq(parent, child, sos_token=to_ix["<sos>"])

            encoder = seq2seq.get_encoder()
            discriminator_encoder.rnn.load_state_dict(encoder.rnn.state_dict())
            discriminator_encoder.embedding.weight = nn.Parameter(encoder.embedding.weight.T)

            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

            #real batch
            child_one_hot_encoded = nn.functional.one_hot(child).float()
            _, state_real = discriminator_encoder(child_one_hot_encoded)
            state_real = state_real.permute((1,0,2))
            state_real = state_real.reshape(state_real.shape[0], discriminator_encoder.hidden_size*2)

            pred_real = discriminator(state_real)      

            #fake batch
            _, state_fake = discriminator_encoder(generated_child.detach())
            state_fake = state_real.permute((1,0,2))
            state_fake = state_fake.reshape(state_fake.shape[0], discriminator_encoder.hidden_size*2)

            pred_fake = discriminator(state_fake)

            #TODO here you add the "not-child" sequences too

            loss_discriminator = torch.mean(pred_real) - torch.mean(pred_fake)
            loss_discriminator.backward(retain_graph=True)

            loss_generator = torch.mean(pred_fake)
            loss_generator.backward()

            optimizer_discriminator.step()
            optimizer_generator.step()
    