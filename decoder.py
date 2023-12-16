import torch
import copy
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
        Q = self.proj_Q(input_tensor).reshape(-1, self.c1)
        K = self.proj_K(input_tensor).reshape(-1, self.c1)
        V = self.proj_V(input_tensor).reshape(-1, self.c1)
        return Q, K, V

class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout_probability, masked, c1):
        super().__init__()
        self.masked = masked
        self.heads_QKV = LinearProjection(d_model, c1=c1)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.attention_weights = None #Cache purposes ex visualization 

    def Attention(self, encodings): 
        ''' MA = softmax(QK_t + M)'''
        query, key, value = self.heads_QKV(encodings)
        d_k = query.size(-1)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if self.masked:
            mask = self.subsequent_mask(attention.size(1))
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

    def forward(self, head_encodings, cache=True):
        attention_weights = self.Attention(head_encodings)
        if cache:
            self.attention_weights = attention_weights
        return attention_weights

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
                                        masked, c1=c1), heads
                                        )
        self.proj_z = nn.Linear(c1, 1) #turn the tensor [n, c1] into a tensor [n, 1]
        self.proj_w = nn.Linear(heads, 1)
    
    def forward(self, encodings):
        for i, head in enumerate(self.selfattention):
            #partial_rapresentation = head(encodings[i*self.d_model:(i+1)*self.d_model])
            partial_rapresentation = head(encodings)
            if i == 0:
                concatenated_rapresentation = partial_rapresentation
            else:
                concatenated_rapresentation = torch.cat((concatenated_rapresentation, partial_rapresentation), dim=0)
        concatenated_rapresentation = self.proj_z(concatenated_rapresentation).transpose(0, 2)
        concatenated_rapresentation = self.proj_w(concatenated_rapresentation)
        return concatenated_rapresentation

class Decoder(nn.Module):
    def __init__(self, heads, d_model, n_out=1, dropout_probability=0.1, masked=True, c1=4, c_ff=4):
        super().__init__()
        self.in_norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_probability)
        self.multiheadattentionlayer = MultiHeadAttention(heads, d_model, dropout_probability, masked, c1)
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
        x = self.out_proj(x)
        return x

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = (inputs, targets)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs[0], targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 0:
                    print(f'Predict: {outputs} while target was {targets}')
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Error: {avg_loss:.4f}")
        return avg_loss


class TimeSeriesData(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        input_sequence = self.data['y'].iloc[idx:idx + self.sequence_length].values
        target = self.data['y'].iloc[idx + self.sequence_length]  # Valore immediatamente successivo

        return torch.tensor(input_sequence, dtype=torch.float), torch.tensor(target, dtype=torch.float)
        

def decoder_debugger(train, test):
    batch = torch.tensor(train['y'][-90:].values, dtype=torch.float32)
    decoder = Decoder(3, 90, c1=90)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
    criterion = F.mse_loss  

    dataset =  TimeSeriesData(train, 90)
    
    trainloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1
                            )
    #iterator = iter(trainloader)
    trainer =  Trainer(decoder, criterion, optimizer)
    trainer.train(trainloader, 10)
    
    print('end')
    
    