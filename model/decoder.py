import torch
import copy
import math
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
torch.manual_seed(0)

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

class SkipConnection(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, residual, current):
        return residual + current

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class LinearProjection(nn.Module):
    def __init__(self, input_size, c1):
        super().__init__()
        self.c1 = c1
        self.proj_Q = nn.Linear(input_size, c1*input_size)
        self.proj_K = nn.Linear(input_size, c1*input_size)
        self.proj_V = nn.Linear(input_size, c1*input_size)

    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        Q = self.proj_Q(input_tensor).reshape(batch_size, 1, -1, self.c1)
        K = self.proj_K(input_tensor).reshape(batch_size, 1, -1, self.c1)
        V = self.proj_V(input_tensor).reshape(batch_size, 1, -1, self.c1)
        return Q, K, V

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout_probability, masked, c1):
        super().__init__()
        self.masked = masked
        self.heads_QKV = LinearProjection(d_model, c1=c1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_probability)

    def Attention(self, encodings): 
        ''' MA = softmax(QK_t + M)'''
        query, key, value = self.heads_QKV(encodings)
        d_k = query.size(-1)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if self.masked:
            mask = self.subsequent_mask(attention.size(-1))
            '''mask could be optional if used in a trasformer architecture'''
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.dropout(self.softmax(attention))
        ''' MSA = MA*V'''
        return torch.matmul(attention, value)

    def subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0

    def forward(self, head_encodings):
        return self.Attention(head_encodings)

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout_probability, masked, c1):
        super().__init__()
        assert d_model % heads == 0, f'Model dimension must be divisible by the number of heads.'
        self.heads = heads
        self.d_model = d_model
        self.d_heads = int(d_model / heads)
        self.selfattention = get_clones(SelfAttention(
                                        self.d_model, 
                                        dropout_probability, 
                                        masked, c1=c1), heads)
        self.proj_z = nn.Linear(c1, 1)
        self.proj_w = nn.Linear(heads, 1)
    
    def forward(self, encodings):
        concatenated_rapresentation = torch.cat([head(encodings) for head in self.selfattention], dim=1)
        concatenated_rapresentation = self.proj_z(concatenated_rapresentation).transpose(1, 3)
        concatenated_rapresentation = self.proj_w(concatenated_rapresentation)
        return concatenated_rapresentation

class Decoder(nn.Module):
    def __init__(self, heads, d_model, n_out=1, dropout_probability=0.1, masked=True, c1=4, c_ff=4):
        super().__init__()
        self.in_norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_probability)
        self.multiheadattentionlayer = MultiHeadAttention(heads, 
                                                          d_model, 
                                                          dropout_probability, 
                                                          masked, 
                                                          c1)
        self.in_skip_connection = SkipConnection()
        self.hid_norm_layer = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model*c_ff, d_model)
        self.hid_skip_connection = SkipConnection()
        self.out_proj = nn.Linear(d_model, n_out) 

    def forward(self, encodings):
        x = self.in_norm_layer(encodings)
        res = self.dropout(x)
        x = self.multiheadattentionlayer(res)
        x = self.in_skip_connection(res, torch.squeeze(x))
        res = self.hid_norm_layer(x)
        x = self.feed_forward(res)
        x = self.hid_skip_connection(res, x)
        x = self.out_proj(x).squeeze(dim=1)
        return x

class Trainer:
    """Addestra il modello; `save` (opzionale) riceve il modello quando la loss di validazione tocca un nuovo
    minimo, o alla fine se non c'è validazione."""
    def __init__(self, model, criterion, optimizer, save=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save = save or (lambda model: None)

    def train(self, train_loader, num_epochs, val_loader=None, verbose=False):
        rows = []
        best = math.inf
        for epoch in range(num_epochs):
            self.model.train()  # a ogni epoca: evaluate() lascia il modello in eval
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 5000 == 0 and verbose:
                    print(f'Predict: {outputs} while target was {targets}')
            epoch_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader) if val_loader is not None else math.nan
            print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f}")
            if val_loss < best:
                best = val_loss
                self.save(self.model)
            rows.append({'epoch': epoch, 't_loss': epoch_loss, 'v_loss': val_loss})
        if val_loader is None:
            self.save(self.model)
        return pd.DataFrame(rows, columns=['epoch', 't_loss', 'v_loss'])

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(val_loader)

class Forecaster:
    def __init__(self, model):
        self.model = model

    def forecast(self, inputs):
        with torch.no_grad():
            output = self.model(inputs)
            inputs = inputs.squeeze()
            next_ar = torch.cat((inputs, output))
            next_ar = next_ar[1:].unsqueeze(0)
        return output, next_ar

    def __call__(self, inputs, horizon):
        next_ar = inputs
        forecast = torch.empty(size=(1, horizon), dtype=torch.float32)
        for i in range(horizon):
            output, next_ar = self.forecast(next_ar)
            forecast[0, i] = output
        return forecast

def decoder_training_pipeline(train_data, valid_data):
    module = utils.utils()
    decoder = Decoder(module.get_heads(), module.get_d_model(), c1=module.get_c1())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=module.get_lr())
    criterion = F.mse_loss  

    train_data =  utils.TimeSeriesData(train_data, module.get_d_model())
    valid_data = utils.TimeSeriesData(valid_data, module.get_d_model())

    trainloader = DataLoader(dataset=train_data,
                            batch_size=module.get_batch_size(),
                            shuffle=False,
                            num_workers=1
                            )
    
    validationloader = DataLoader(dataset=valid_data,
                                batch_size=module.get_batch_size(),
                                shuffle=False,
                                num_workers=1
                                )
    trainer = Trainer(decoder, criterion, optimizer, save=module.save)
    loss_over_epoch = trainer.train(trainloader, module.get_epochs(), validationloader)
    return loss_over_epoch
    
def decoder_forecasting_pipeline(starting_window, horizon):
    module = utils.utils()
    decoder = Decoder(module.get_heads(), module.get_d_model(), c1=module.get_c1())
    decoder = module.load(decoder)
    decoder.eval()
    forecaster = Forecaster(decoder)
    transformed_window = starting_window.unsqueeze(0)
    forecast = forecaster(transformed_window, horizon)
    return forecast
