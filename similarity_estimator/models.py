from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import os


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, opt, is_train=False, pretrained_embeddings=None):
        super(BiLSTM, self).__init__()
        self.opt = opt
        self.drop = nn.Dropout(self.opt.dropout)
        self.encoder = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.opt.embedding_dims,
                                    padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self.bilstm = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims, num_layers=1,
                              bidirectional=False)
        self.nlayers = self.opt.num_layers
        if pretrained_embeddings is not None:
            self.encoder.weight.data.copy_(pretrained_embeddings)
        self.nhid = self.opt.hidden_dims
        self.pooling = self.opt.pooling
        self.dictionary = ' '  # location of
        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['ninp']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]
        if self.opt.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.opt.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.opt.pooling == 'all' or self.opt.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, vocab_size, opt, is_train=True):
        super(SelfAttentiveEncoder, self).__init__()
        self.opt = opt
        self.bilstm = BiLSTM(vocab_size, self.opt, is_train)
        self.drop = nn.Dropout(self.opt.dropout)
        self.ws1 = nn.Linear(self.opt.hidden_dims * 2, self.opt.da, bias=False)
        self.ws2 = nn.Linear(self.opt.da, self.opt.r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        # self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = self.opt.r

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        # penalized_alphas = alphas + (
        #     -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']
#        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        if type(self.encoder) == BiLSTM:
            attention = None
        return pred, attention

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]
