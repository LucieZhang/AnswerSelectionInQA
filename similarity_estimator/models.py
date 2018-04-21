"""copy of the original networks"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

# Inference with SVR
import pickle

from utils.parameter_initialization import xavier_normal


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, opt, is_train=False):
        super(LSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.opt = opt
        self.name = 'sentence representation'

        self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.opt.embedding_dims,
                                            padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        self.lstm_rnn = nn.LSTM(input_size=self.opt.embedding_dims, hidden_size=self.opt.hidden_dims, num_layers=1,
                                bidirectional=False, bias=False, batch_first=True)
        # Self Attention Layers
        self.S1 = nn.Linear(self.opt.hidden_dims, self.opt.da, bias=False)
        self.S2 = nn.Linear(self.opt.da, self.opt.r, bias=False)

        self.init_weights()


    def initialize_hidden_plus_cell(self, batch_size):

        if torch.cuda.is_available():
            zero_hidden = Variable(torch.randn(1, batch_size, self.opt.hidden_dims).cuda(), requires_grad=True)
            zero_cell = Variable(torch.randn(1, batch_size, self.opt.hidden_dims).cuda(), requires_grad=True)
        else:
            zero_hidden = Variable(torch.randn(1, batch_size, self.opt.hidden_dims), requires_grad=True)
            zero_cell = Variable(torch.randn(1, batch_size, self.opt.hidden_dims), requires_grad=True)

        return zero_hidden, zero_cell

    def init_weights(self):
        initrange = 0.1
        self.S1.weight.data.uniform_(-initrange, initrange)
        self.S2.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch_size, input_data, hidden, cell, is_question_embed=True):

        if is_question_embed:
            input_data = torch.transpose(input_data, 0, 1)
            output = self.embedding_table(input_data)
        else:
            output = self.embedding_table(input_data).view(batch_size, 1, -1)
        for _ in range(self.opt.num_layers):
            output, (hidden, cell) = self.lstm_rnn(output, (hidden, cell))

        if is_question_embed:
            if torch.cuda.is_available():
                BM = Variable(torch.zeros(input_data.size(0), self.opt.r * self.opt.hidden_dims).cuda())
                penal = Variable(torch.zeros(1).cuda())
                I = Variable(torch.eye(self.opt.r).cuda())
            else:
                BM = Variable(torch.zeros(input_data.size(0), self.opt.r * self.opt.hidden_dims))
                penal = Variable(torch.zeros(1))
                I = Variable(torch.eye(self.opt.r))
            weights = {}

            # Attention
            for i in range(input_data.size(0)):  # for i in batch_size
                H = output[i, :input_data.size(1), :]  # output[bs, sequence, hidden_dim] for question

                s1 = self.S1(H)
                s2 = self.S2(functional.tanh(s1))

                A = functional.softmax(s2.t())
                M = torch.mm(A, H)  # M is the sentence embedding, r * 2u
                BM[i, :] = M.view(-1)

                # Penalization
                AAT = torch.mm(A, A.t())
                P = torch.norm(AAT - I, 2)
                penal += P * P
                weights[i] = A

            # Penalization
            penal /= input_data.size(0)

            return BM, hidden, cell, penal
        else:
            return output, hidden, cell


class AnswerSelection(nn.Module):

    def __init__(self, vocab_size, opt, pretrained_embeddings=None, is_train=True):
        super(AnswerSelection, self).__init__()
        self.opt = opt
        self.is_train = is_train

        self.encoder_a = self.encoder_b = self.encoder_c = LSTMEncoder(vocab_size, self.opt, is_train)
        if pretrained_embeddings is not None:
            self.encoder_a.embedding_table.weight.data.copy_(pretrained_embeddings)
            self.encoder_b.embedding_table.weight.data.copy_(pretrained_embeddings)
            self.encoder_c.embedding_table.weight.data.copy_(pretrained_embeddings)
        self.initialize_parameters()

        self.loss_function = nn.MarginRankingLoss(margin=0.5)
        self.optimizer_a = optim.Adam(self.encoder_a.parameters(), lr=self.opt.learning_rate,
                                      betas=(self.opt.beta_1, 0.999))
        self.optimizer_b = optim.Adam(self.encoder_b.parameters(), lr=self.opt.learning_rate,
                                      betas=(self.opt.beta_1, 0.999))
        self.optimizer_c = optim.Adam(self.encoder_c.parameters(), lr=self.opt.learning_rate,
                                      betas=(self.opt.beta_1, 0.999))
        self.distance = nn.PairwiseDistance(1)

    def forward(self):
        state_dict = self.encoder_b.state_dict()
        # input length (each batch consists of padded sentences)
        input_length = self.batch_a.size(0)

        hidden_a, cell_a = self.encoder_a.initialize_hidden_plus_cell(self.batch_size)
        # for t_i in range(input_length):
        #     output_a, hidden_a, cell_a = self.encoder_a(self.batch_size, self.batch_a[t_i, :], hidden_a, cell_a)
        # output_a dimension: 1 * 2u
        output_a, hidden_a, cell_a, self.penal = self.encoder_a(self.batch_size, self.batch_a, hidden_a, cell_a,
                                                            is_question_embed=True)

        # weight-sharing
        # self.encoder_b.load_state_dict(state_dict)
        hidden_b, cell_b = self.encoder_b.initialize_hidden_plus_cell(self.batch_size)
        for t_j in range(input_length):
            output_b, hidden_b, cell_b = self.encoder_b(self.batch_size, self.batch_b[t_j, :], hidden_b, cell_b,
                                                        is_question_embed=False)

        if self.is_train:
            self.encoder_c.load_state_dict(state_dict)
            hidden_c, cell_c = self.encoder_c.initialize_hidden_plus_cell(self.batch_size)
            for t_k in range(input_length):
                output_c, hidden_c, cell_c = self.encoder_c(self.batch_size, self.batch_c[t_k, :], hidden_c, cell_c,
                                                            is_question_embed=False)

        # Format sentence encodings as 2D tensors
        self.encoding_a = output_a
        self.encoding_b = hidden_b.squeeze()
        if self.is_train:
            self.encoding_c = hidden_c.squeeze()

            if self.batch_size == 1:
                self.prediction1 = self.distance(self.encoding_a.view(1, -1), self.encoding_b.view(1, -1)).view(-1, 1)
                self.prediction2 = self.distance(self.encoding_a.view(1, -1), self.encoding_c.view(1, -1)).view(-1, 1)
            else:
                self.prediction1 = self.distance(self.encoding_a, self.encoding_b).view(-1, 1)
                self.prediction2 = self.distance(self.encoding_a, self.encoding_c).view(-1, 1)
            return self.prediction1,self.prediction2
        # self.prediction = self.tan(self.sim_score)
        else:
            return

    # get the similarity score of a qa pair
    def get_score(self):
        if self.batch_size == 1:
            self.score = self.distance(self.encoding_a.view(1, -1), self.encoding_b.view(1, -1)).view(-1, 1)
        else:
            self.score = self.distance(self.encoding_a, self.encoding_b).view(-1, 1)

        return self.score

    def get_loss(self):

        # if self.batch_size == 1:
        # self.loss = self.loss_function(self.encoding_a.view(1, -1), self.encoding_b.view(1, -1), self.labels)
        # else:
        # self.loss = self.loss_function(self.encoding_a, self.encoding_b, self.labels)
        self.loss = self.loss_function(self.prediction2, self.prediction1, self.labels) + self.penal

    def load_pretrained_parameters(self):

        pretrained_state_dict_path = os.path.join(self.opt.pretraining_dir, self.opt.pretrained_state_dict)
        self.encoder_a.load_state_dict(torch.load(pretrained_state_dict_path))
        print('Pretrained parameters have been successfully loaded into the encoder networks.')

    def initialize_parameters(self):

        state_dict = self.encoder_a.state_dict()
        for key in state_dict.keys():
            if '.weight' in key:
                state_dict[key] = xavier_normal(state_dict[key])
            if '.bias' in key:
                bias_length = state_dict[key].size()[0]
                start, end = bias_length // 4, bias_length // 2
                state_dict[key][start:end].fill_(2.5)
        self.encoder_a.load_state_dict(state_dict)

    def train_step(self, train_batch_a, train_batch_b, train_batch_c,
                   train_labels):
        if torch.cuda.is_available():
            self.batch_a = train_batch_a.cuda()
            self.batch_b = train_batch_b.cuda()
            self.batch_c = train_batch_c.cuda()
            self.labels = train_labels.cuda()
        else:
            self.batch_a = train_batch_a
            self.batch_b = train_batch_b
            self.batch_c = train_batch_c
            self.labels = train_labels

        self.batch_size = self.batch_a.size(1)

        self.forward()
        self.encoder_a.zero_grad()  # encoder_a == encoder_b
        self.encoder_b.zero_grad()
        self.encoder_c.zero_grad()
        self.get_loss()
        self.loss.backward()
        clip_grad_norm(self.encoder_a.parameters(), self.opt.clip_value)

        self.optimizer_a.step()
        self.optimizer_b.step()
        self.optimizer_c.step()
        return self.prediction1,self.prediction2

    def test_step(self, test_batch_a, test_batch_b, test_batch_c, test_labels):
        if torch.cuda.is_available():
            self.batch_a = test_batch_a.cuda()
            self.batch_b = test_batch_b.cuda()
            self.batch_c = test_batch_c.cuda()
            self.labels = test_labels.cuda()
        else:
            self.batch_a = test_batch_a
            self.batch_b = test_batch_b
            self.batch_c = test_batch_c
            self.labels = test_labels

        self.batch_size = self.batch_a.size(1)
        #
        self.forward()
        self.get_loss()
        return self.prediction1, self.prediction2

    def qa_step(self, test_batch_a, test_batch_b, test_labels):
        self.is_train = False

        if torch.cuda.is_available():
            self.batch_a = test_batch_a.cuda()
            self.batch_b = test_batch_b.cuda()
            self.labels = test_labels.cuda()
        else:
            self.batch_a = test_batch_a
            self.batch_b = test_batch_b
            self.labels = test_labels

        self.batch_size = self.batch_a.size(1)
        self.forward()

