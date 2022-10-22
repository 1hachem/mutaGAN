import torch
import torch.nn as nn
from tqdm import tqdm

from src.encode import to_ix


def GAN_train(seq2seq, discriminator_encoder, discriminator, optimizer_generator, optimizer_discriminator, parent_data_loader, child_data_loader, not_child_data_loader, num_epochs, device, neptune_run=None):
    for epoch in range(num_epochs):
        progress_bar = tqdm(zip(parent_data_loader, child_data_loader, not_child_data_loader), total=len(parent_data_loader))
        for parent, child, not_child in progress_bar:

            parent = parent.to(device)
            child = child.to(device)
            not_child = not_child.to(device)

            generated_child, state_parent = seq2seq(parent, child, sos_token=to_ix["<sos>"])
            state_parent = state_parent.permute((1,0,2))
            state_parent = state_parent.reshape(state_parent.shape[0], discriminator_encoder.num_layers*discriminator_encoder.hidden_size*2)

            encoder = seq2seq.get_encoder()
            discriminator_encoder.rnn.load_state_dict(encoder.rnn.state_dict())
            discriminator_encoder.embedding.weight = nn.Parameter(encoder.embedding.weight.T)

            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()
            
            #real batch
            child_one_hot_encoded = nn.functional.one_hot(child).float()
            _, state_real = discriminator_encoder(child_one_hot_encoded)
            state_real = state_real.permute((1,0,2))
            state_real = state_real.reshape(state_real.shape[0], discriminator_encoder.num_layers*discriminator_encoder.hidden_size*2)

            pred_real = discriminator(state_parent, state_real)      

            #fake batch
            _, state_fake = discriminator_encoder(generated_child.detach())
            state_fake = state_fake.permute((1,0,2))
            state_fake = state_fake.reshape(state_fake.shape[0], discriminator_encoder.num_layers*discriminator_encoder.hidden_size*2)

            pred_fake = discriminator(state_parent, state_fake)

            #not-child batch
            not_child_one_hot_encoded = nn.functional.one_hot(not_child).float()
            _, state_not_child = discriminator_encoder(not_child_one_hot_encoded)
            state_not_child = state_not_child.permute((1,0,2))
            state_not_child = state_not_child.reshape(state_not_child.shape[0], discriminator_encoder.num_layers*discriminator_encoder.hidden_size*2)

            pred_not_child = discriminator(state_parent, state_not_child)

            loss_discriminator = torch.mean(pred_real) - torch.mean(pred_fake) - torch.mean(pred_not_child) #TODO write it well 
            loss_discriminator.backward(retain_graph=True)
            cross_loss = nn.CrossEntropyLoss(ignore_index= to_ix["<pad>"])
            loss_generator = torch.mean(pred_fake) + cross_loss(generated_child, child[:,:generated_child.shape[-1]]) #w loss + categorical cross entropy
            loss_generator.backward()

            optimizer_discriminator.step()
            optimizer_generator.step()

            try:
                neptune_run["train_GAN/loss_G"].log(loss_generator.item())
                neptune_run["train_GAN/loss_D"].log(loss_discriminator.item())
            except:
                progress_bar.set_description("error connecting to neptune")
