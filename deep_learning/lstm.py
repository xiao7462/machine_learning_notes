# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
# 
# disable progress bars when submitting
def is_interactive():
   return 'SHLVL' not in os.environ

if not is_interactive():
    def nop(it, *a, **k):
        return it

    tqdm = nop

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


CRAWL_EMBEDDING_PATH = 'crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = 'glove.840B.300d.txt'
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS # 512
MAX_LEN = 220  # 一段话最大的长度


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))



def build_matrix(word_index, path):
    embedding_index = load_embeddings(path) # 返回的embedding_index  是一个len(embedding_index) , 300个维度的 字典    
    embedding_matrix = np.zeros((len(word_index) + 1, 300)) # The word index starts at 1. Zero denotes "not a word" and is used to pad the sequences. But it is not part of the word index. So the length of our embedding matrix has to be one plus the length of the word index.
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)
    # 举例说明
    # optim.Adam([
    #             {'params': model.base.parameters()},
    #             {'params': model.classifier.parameters(), 'lr': 1e-3}
    #         ], lr=1e-2, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train() # 针对训练的  model.eval()针对测试
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)#  *x_batch == x_batch[0]
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        test_preds = np.zeros((len(test), output_dim))
    
        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds

class SpatialDropout(nn.Dropout2d):
#     They describe the current shape of the tensor:

# N: sample dimension (equal to the batch size)
# T: time dimension (equal to MAX_LEN)
# K feature dimension (equal to 300 because of the 300d embeddings)

# nn.Dropout2d randomly zeros out some channels (2nd dimension of a tensor), which have to be the features for spatial dropout. So we have to squeeze and permute the tensor to make the 2nd dimension the feature dimension.
# Dropout2d的input是 Input: (N, C, H, W)(N,C,H,W)
# output是 Output: (N, C, H, W) (same shape as input) 
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1] # embed_size =600
        
        self.embedding = nn.Embedding(max_features, embed_size)  # max_features 是训练的数据集的量 max_features = len(tokenizer.word_index) + 1 = 327576
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3) # 0.3 probability of an element to be zero-ed.
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True) # LSTM_UNITS = 128 
        # 为什么 LSTM_UNITS * 2 https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version/comments#514069
        #It is bidirectional. The lstm1 actually consists of 2 LSTMs, each with LSTM_UNITS units. One is processing the text in forward direction, one going backwards. Thats a common practice when processing text. You can read more about this here.

        #The activations of both LSTMs are then concatenated so the input dimension of the next LSTM becomes LSTM_UNITS * 2
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out

def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data



if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    x_train = preprocess(train['comment_text'])
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    x_test = preprocess(test['comment_text'])


    max_features = None

    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    max_features = len(tokenizer.word_index) + 1
    max_features

    crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
    print('n unknown words (crawl): ', len(unknown_words_crawl))

    glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
    print('n unknown words (glove): ', len(unknown_words_glove))

    embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
    embedding_matrix.shape

    del crawl_matrix
    del glove_matrix
    gc.collect()


    x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
    x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
    y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()


    ## Training

    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    test_dataset = data.TensorDataset(x_test_torch)
    all_test_preds = []

    for model_idx in range(NUM_MODELS):
        print('Model ', model_idx)
        seed_everything(1234 + model_idx)
        
        model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])
        model#.cuda()
        
        test_preds = train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], 
                                 loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
        all_test_preds.append(test_preds)
        print()
    submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': np.mean(all_test_preds, axis=0)[:, 0]
        })
    submission.to_csv('submission.csv', index=False)