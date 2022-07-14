
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, disc=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if disc==True:
            self.embedding = nn.Linear(input_size, embedding_size, bias=False)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding_size)

        self.rnn = nn.LSTM(input_size = embedding_size, hidden_size= self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)


    def forward(self, x):
        #x shape : (batch_size, seq_len)
        #embedding shape : (batch_size, seq_len, embedding_size)

        embedding = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedding)

        return hidden, cell 


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Decoder,self).__init__()
        self.device = device
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size 

        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.embedding_size)
        self.rnn = nn.LSTM(input_size = embedding_size, hidden_size= self.hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, cell):
        embedding = self.embedding(x)
        output, (hidden_, cell_) = self.rnn(embedding, (hidden, cell))
        logits = self.fc(output)

        pred = nn.functional.softmax(logits, dim=-1)

        return pred, hidden_, cell_
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing_ratio=0.7, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Seq2Seq,self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder 
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, parent_batch, child_batch, sos_token):
        batch_size = parent_batch.shape[0]
        child_len = child_batch.shape[-1]

        hidden, state = self.encoder(parent_batch)

        hidden_ = hidden.view(batch_size, self.encoder.hidden_size*2).unsqueeze(0)
        state_ = state.view(batch_size, self.encoder.hidden_size*2).unsqueeze(0)

        #add noise to the state 
        outputs = torch.zeros((batch_size, 1, self.decoder.output_size)).to(self.device)

        #sos_token = to_ix["<sos>"]
        x = sos_token*torch.ones((batch_size,1)).int().to(self.device)

        for t in range(50): #range(child_len):
            #output shape (batch_size, 1, vocab size)
            #outputs shape ((batch_size, seq_len, vocab size))
            output, hidden_, state_ = self.decoder(x, hidden_, state_)
            outputs = torch.cat((outputs, output), dim=1)
            
            #teacher forcing 
            x = child_batch[:,t].unsqueeze(-1).int() if random.random() < self.teacher_forcing_ratio else output.argmax(-1)

        return outputs

    def get_encoder(self):
        return self.encoder

    def set_teacher_forcing_ratio(self, new_ratio):
        self.teacher_forcing_ratio = new_ratio


class Discriminator(nn.Module):
    def __init__(self, hidden_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(2*self.hidden_size),
            nn.Linear(2*self.hidden_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64,1)
            )
        
    def forward(self, x):
        x = self.classifier(x)
        return x