""" Word-based, for now. Switch to sub-word eventually, o a combination of character- and word-based input."""

import random

import torch
from torch.autograd import Variable

from similarity_estimator.sim_util import perform_bucketing


class DataServer(object):
    """ Iterates through a data source, i.e. a list of sentences or list of buckets containing sentences of similar length.
    Produces batch-major batches, i.e. of shape=[seq_len, batch_size]. """
    def __init__(self, data, vocab, vocab2, opt, is_train=False, shuffle=False, use_buckets=True, volatile=False):
        self.data = data
        self.vocab2 = vocab2
        self.vocab = vocab
        self.opt = opt
        self.volatile = volatile
        self.use_buckets = use_buckets
        self.pair_id = 0
        self.buckets = None
        # Obtain bucket data
        if self.use_buckets:
            self.buckets, self.data, self.max_lens_char = perform_bucketing(self.opt, self.data, self.vocab2, is_train=is_train)
            self.bucket_id = 0
        # Select appropriate batch size
        if is_train:
            self.is_train = True
            self.batch_size = self.opt.train_batch_size
        else:
            self.is_train = False
            self.batch_size = self.opt.test_batch_size
        # Shuffle data (either batch-wise or as a whole)
        if shuffle:
            if self.use_buckets:
                # Shuffle within buckets
                for i in range(len(self.data)):
                    zipped = list(zip(*self.data[i]))
                    random.shuffle(zipped)
                    self.data[i] = list(zip(*zipped))
                # Shuffle buckets, also
                bucket_all = list(zip(self.buckets, self.data))
                random.shuffle(bucket_all)
                self.buckets, self.data = zip(*bucket_all)
            else:
                zipped = list(zip(*self.data))
                random.shuffle(zipped)
                self.data = list(zip(*zipped))

    def sent_to_idx(self, sent, vocab, is_char=False):

        idx_list = [vocab.word_to_index[word] if word in vocab.word_to_count.keys() and
                    vocab.word_to_count[word] >= self.opt.freq_bound else 1 for word in sent.split()]
        # Pad to the desired sentence length
        if self.opt.pad:
            if self.use_buckets:
                # Pad to bucket upper length bound
                if is_char:
                    # max_len = self.max_lens_char[self.bucket_id]
                    max_len = self.vocab2.target_len
                else:
                    max_len = self.buckets[self.bucket_id][1]
            else:
                # In case of no bucketing, pad all corpus sentence to a uniform, specified length
                max_len = vocab.target_len
            # Adjust padding for single sentence-pair evalualtion (i.e. no buckets, singleton batches)
            if self.batch_size == 1:
                if self.is_train:
                    if is_char:
                        max_len = max(len(self.separate_chars(self.data[0][self.pair_id][0]).split()), len(self.separate_chars(self.data[0][self.pair_id][1]).split()),
                                      len(self.separate_chars(self.data[0][self.pair_id][2]).split()))
                    else:
                        max_len = max(len(self.data[0][self.pair_id][0].split()), len(self.data[0][self.pair_id][1].split()),
                                len(self.data[0][self.pair_id][2].split()))
                else:
                    if is_char:
                        max_len = max(len(self.separate_chars(self.data[0][self.pair_id][0]).split()),
                                      len(self.separate_chars(self.data[0][self.pair_id][1]).split()))
                    else:
                        max_len = max(len(self.data[0][self.pair_id][0].split()), len(self.data[0][self.pair_id][1].split()))
            # Pad items to maximum length
            diff = max_len - len(idx_list)
            if diff >= 1:
                idx_list += [0] * diff
        return idx_list

    def __iter__(self):
        """ Returns an iterator object. """
        return self

    def __next__(self):
        """ Returns the next batch from within the iterator source. """
        try:
            if self.use_buckets:
                if self.is_train:
                    if self.vocab2 is not None:
                        s1_batch, s2_batch, s3_batch, c1_batch, c2_batch, c3_batch, label_batch = self.bucketed_next()
                    else:
                        s1_batch, s2_batch, s3_batch, label_batch = self.bucketed_next()
                else:
                    if self.vocab2 is not None:
                        s1_batch, s2_batch, c1_batch, c2_batch, label_batch = self.bucketed_next()
                    else:
                        s1_batch, s2_batch, label_batch = self.bucketed_next()
            else:
                if self.is_train:
                    if self.vocab2 is not None:
                        s1_batch, s2_batch, s3_batch, c1_batch, c2_batch, c3_batch, label_batch = self.corpus_next()
                    else:
                        s1_batch, s2_batch, s3_batch, label_batch = self.corpus_next()
                else:
                    if self.vocab2 is not None:
                        s1_batch, s2_batch, c1_batch, c2_batch, label_batch = self.corpus_next()
                    else:
                        s1_batch, s2_batch, label_batch = self.corpus_next()
        except IndexError:
            raise StopIteration

        # Covert batches into batch major form
        s1_batch = torch.LongTensor(s1_batch).t().contiguous()
        s2_batch = torch.LongTensor(s2_batch).t().contiguous()
        if self.vocab2 is not None:
            c1_batch = torch.LongTensor(c1_batch).t().contiguous()
            c2_batch = torch.LongTensor(c2_batch).t().contiguous()

            c1_var = Variable(c1_batch, volatile=self.volatile)
            c2_var = Variable(c2_batch, volatile=self.volatile)
        if self.is_train:
            s3_batch = torch.LongTensor(s3_batch).t().contiguous()
            if self.vocab2 is not None:
                c3_batch = torch.LongTensor(c3_batch).t().contiguous()
                c3_var = Variable(c3_batch, volatile=self.volatile)
        label_batch = torch.FloatTensor(label_batch).contiguous()

        # print('the length of s1_batch = [%d]' % len(s1_batch))
        # Convert to variables
        s1_var = Variable(s1_batch, volatile=self.volatile)
        s2_var = Variable(s2_batch, volatile=self.volatile)
        if self.is_train:
            s3_var = Variable(s3_batch, volatile=self.volatile)
        label_var = Variable(label_batch, volatile=self.volatile)
        if self.is_train:
            if self.vocab2 is not None:
                return s1_var, s2_var, s3_var, c1_var, c2_var, c3_var, label_var
            else:
                return s1_var, s2_var, s3_var, label_var
        else:
            if self.vocab2 is not None:
                return s1_var, s2_var, c1_var, c2_var, label_var
            else:
                return s1_var, s2_var, label_var

    def separate_chars(self, sent):
        sent_list = list(sent)
        char_list = []
        for i, ch in enumerate(sent_list):
            if ch != u" ":
                char_list.append(ch + u" ")
        sent_char = "".join(char_list[:self.vocab2.target_len])
        return sent_char

    def bucketed_next(self):
        """ Samples the next batch from the current bucket. """
        # Assemble batches
        s1_batch = list()
        s2_batch = list()
        s3_batch = list()
        c1_batch = list()
        c2_batch = list()
        c3_batch = list()
        label_batch = list()

        if self.bucket_id < self.opt.num_buckets:
            # Fill batches
            while len(s1_batch) < self.batch_size:
                try:
                    s1 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][0], self.vocab)
                    s2 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][1], self.vocab)
                    if self.vocab2 is not None:
                        s1_ch = self.separate_chars(self.data[self.bucket_id][0][self.pair_id][0])
                        s2_ch = self.separate_chars(self.data[self.bucket_id][0][self.pair_id][1])
                        # max_len = max(len(s1_ch.split()), len(s2_ch.split()))
                        if self.is_train:
                            s3 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][2], self.vocab)
                            s3_ch = self.separate_chars(self.data[self.bucket_id][0][self.pair_id][2])
                            # max_len = max(max_len, len(s3_ch.split()))
                            c3 = self.sent_to_idx(s3_ch, self.vocab2, is_char=True)
                        c1 = self.sent_to_idx(s1_ch, self.vocab2, is_char=True)
                        c2 = self.sent_to_idx(s2_ch, self.vocab2, is_char=True)
                    elif self.is_train:
                        s3 = self.sent_to_idx(self.data[self.bucket_id][0][self.pair_id][2], self.vocab)

                    label = [float(self.data[self.bucket_id][1][self.pair_id])]
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    if self.vocab2 is not None:
                        c1_batch.append(c1)
                        c2_batch.append(c2)
                    if self.is_train:
                        s3_batch.append(s3)
                        if self.vocab2 is not None:
                            c3_batch.append(c3)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    # Finish batch prematurely if bucket or corpus has been emptied
                    self.pair_id = 0
                    self.bucket_id += 1
                    break
            # Check if bucket is empty, to avoid generation of empty batches
            try:
                if self.pair_id == len(self.data[self.bucket_id][0]):
                    self.bucket_id += 1
            except IndexError:
                pass
        else:
            raise IndexError

        if self.is_train:
            if self.vocab2 is not None:
                return s1_batch, s2_batch, s3_batch, c1_batch, c2_batch, c3_batch, label_batch
            else:
                return s1_batch, s2_batch, s3_batch, label_batch
        else:
            if self.vocab2 is not None:
                return s1_batch, s2_batch, c1_batch, c2_batch, label_batch
            else:
                return s1_batch, s2_batch, label_batch

    def corpus_next(self):
        """ Samples the next batch from the un-bucketed corpus. """
        # Assemble batches
        s1_batch = list()
        s2_batch = list()
        s3_batch = list()
        c1_batch = list()
        c2_batch = list()
        c3_batch = list()
        label_batch = list()

        # Without bucketing
        if self.pair_id < self.get_length():
            while len(s1_batch) < self.batch_size:
                try:
                    s1 = self.sent_to_idx(self.data[0][self.pair_id][0], self.vocab)
                    s2 = self.sent_to_idx(self.data[0][self.pair_id][1], self.vocab)
                    if self.vocab2 is not None:
                        s1_ch = self.separate_chars(self.data[0][self.pair_id][0])
                        s2_ch = self.separate_chars(self.data[0][self.pair_id][1])
                        c1 = self.sent_to_idx(s1_ch, self.vocab2, is_char=True)
                        c2 = self.sent_to_idx(s2_ch, self.vocab2, is_char=True)
                    if self.is_train:
                        s3 = self.sent_to_idx(self.data[0][self.pair_id][2], self.vocab)
                        if self.vocab2 is not None:
                            s3_ch = self.separate_chars(self.data[0][self.pair_id][2])
                            c3 = self.sent_to_idx(s3_ch, self.vocab2, is_char=True)
                    label = [(self.data[1][self.pair_id])]  # float
                    s1_batch.append(s1)
                    s2_batch.append(s2)
                    if self.vocab2 is not None:
                        c1_batch.append(c1)
                        c2_batch.append(c2)
                    if self.is_train:
                        s3_batch.append(s3)
                        if self.vocab2 is not None:
                            c3_batch.append(c3)
                    label_batch.append(label)
                    self.pair_id += 1
                except IndexError:
                    break
        else:
            raise IndexError

        if self.is_train:
            if self.vocab2 is not None:
                return s1_batch, s2_batch, s3_batch, c1_batch, c2_batch, c3_batch, label_batch
            else:
                return s1_batch, s2_batch, s3_batch, label_batch
        else:
            if self.vocab2 is not None:
                return s1_batch, s2_batch, c1_batch, c2_batch, label_batch
            else:
                return s1_batch, s2_batch, label_batch

    def get_length(self):
        # Return corpus length in sentence pairs
        if self.use_buckets:
            return sum([len(bucket[0]) for bucket in self.data])
        else:
            return len(self.data[0])
