import numpy as np
import pandas as pd


class Indexer(object):

    def __init__(self, name):
        self.name = name
        self.word_to_index = dict()
        self.word_to_count = dict()
        # Specify start-and-end-of-sentence tokens
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.n_words = 2

        self.target_len = None

    def add_sentence(self, sentence):

        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):

        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.index_to_word[self.n_words] = word
            self.word_to_count[word] = 1
            self.n_words += 1
        else:
            self.word_to_count[word] += 1

    # def add_char(self, word):

    def set_target_len(self, value):
        self.target_len = value


def perform_bucketing(opt, labeled_pair_list, is_train=True):
    # Groups the provided sentence pairs into the specified number of buckets of similar size based on the length of
    # their longest member.
    # Obtain sentence lengths
    if is_train:
        sentence_pair_lens = [(len(pair[0].split()), len(pair[1].split()), len(pair[2].split()))
                            for pair in labeled_pair_list[0]]
    else:
        sentence_pair_lens = [(len(pair[0].split()), len(pair[1].split()))
                              for pair in labeled_pair_list[0]]

    # Calculate bucket size
    buckets = [[0, 0] for _ in range(opt.num_buckets)]
    avg_bucket = len(labeled_pair_list[0]) // opt.num_buckets  # number of data cases
    if is_train:
        max_lens = [max(pair[0], pair[1], pair[2]) for pair in sentence_pair_lens]
    else:
        max_lens = [max(pair[0], pair[1]) for pair in sentence_pair_lens]
    len_counts = [(sent_len, max_lens.count(sent_len)) for sent_len in set(max_lens)]
    len_counts.sort(key=lambda x: x[0])

    bucket_pointer = 0
    len_pointer = 0

    while bucket_pointer < opt.num_buckets and len_pointer < len(len_counts):
        target_bucket = buckets[bucket_pointer]
        # Set lower limit on the bucket's lengths
        target_bucket[0] = len_counts[len_pointer][0]
        bucket_load = 0
        while True:
            try:
                len_count_pair = len_counts[len_pointer]
                deficit = avg_bucket - bucket_load
                surplus = (bucket_load + len_count_pair[1]) - avg_bucket
                if deficit >= surplus or bucket_pointer == opt.num_buckets - 1:
                    bucket_load += len_count_pair[1]
                    # Update upper limit on the bucket's lengths
                    target_bucket[1] = len_count_pair[0]
                    len_pointer += 1
                else:
                    bucket_pointer += 1
                    break
            except IndexError:
                break

    # Populate buckets
    bucketed = [([], []) for _ in range(opt.num_buckets)]
    for k in range(len(labeled_pair_list[0])):
        if is_train:
            pair_len = max(sentence_pair_lens[k][0], sentence_pair_lens[k][1], sentence_pair_lens[k][2])
        else:
            pair_len = max(sentence_pair_lens[k][0], sentence_pair_lens[k][1])
        for l in range(len(buckets)):
            if buckets[l][0] <= pair_len <= buckets[l][1]:
                bucketed[l][0].append(labeled_pair_list[0][k])
                bucketed[l][1].append(labeled_pair_list[1][k])

    return buckets, bucketed


def load_similarity_data(opt, corpus_location, corpus_name, is_train=True, is_dbqa=False):

    if is_train:
        df_sim = pd.read_table(corpus_location, sep='\t', header=None, encoding='utf-8', error_bad_lines=False,
                               names=['question', 'triple1', 'triple2', 'label'], skip_blank_lines=True, engine='python', nrows=500)
    else:
        df_sim = pd.read_table(corpus_location, sep='\t', header=None, encoding='utf-8', error_bad_lines=False,
                               names=['question', 'triple1', 'label'], skip_blank_lines=True, engine='python', nrows=1010)

    sim_data = [[], []]
    sim_sents = list()
    # get sentence lengths for max and mean length calculations
    sent_lens = list()
    for i in range(len(df_sim['label'])):

        sent_question = df_sim.iloc[i, 0].strip()
        sent_triple1 = df_sim.iloc[i, 1].strip()
        if is_train:
            sent_triple2 = df_sim.iloc[i, 2].strip()

        if is_dbqa:
            sent_a = str(sent_question)
            sent_b = str(sent_triple1)
            if is_train:
                sent_b2 = str(sent_triple2)
        else:
            try:
                sent_sub1 = sent_triple1[:sent_triple1.index(' |||')]
                sent_pre_ans1 = sent_triple1[sent_triple1.index(' ||| ') + 5:]  # 去掉三元组主语
                if is_train:
                    sent_pre_ans2 = sent_triple2[sent_triple2.index(' |||') + 5:]
                    sent_b2 = sent_pre_ans2[:sent_pre_ans2.index(' |||')]
                sent_a = sent_question
                sent_b = sent_pre_ans1[:sent_pre_ans1.index(' |||')]  # 提取主语+谓词/去掉主语
            except ValueError:
                print('skip:\t' + sent_question + '\t' + sent_triple1)
                continue

        # len_sub = len(sent_sub)
        # if is_train:
        #     if sent_sub in sent_question:
        #         sent_a = sent_question[sent_question.index(sent_sub) + len_sub + 1:]
        #     else:
        #         sent_a = sent_question[len_sub + 1:]
        #     sent_b = sent_pre_ans[:sent_pre_ans.index(' |||')]  # 提取谓词
        # else:
        #     sent_a = sent_question
        #     sent_b = sent_triple

        # label = "{:.4f}".format(float(df_qa.iloc[i, 2]))
        if is_train:
            label = int(df_sim.iloc[i, 3].replace(' ', ''))
        else:
            label = int(df_sim.iloc[i, 2].replace(' ', ''))

        # Assemble a list of tuples containing the compared sentences, and track the max observed length
        if is_train:
            sim_data[0].append((sent_a, sent_b, sent_b2))
            sim_sents += [sent_a, sent_b, sent_b2]
            sent_lens += [len(sent_a.split()), len(sent_b.split()), len(sent_b2.split())]
        else:
            sim_data[0].append((sent_a, sent_b))
            sim_sents += [sent_a, sent_b]
            sent_lens += [len(sent_a.split()), len(sent_b.split())]
        sim_data[1].append(label)

    # Filter corpus according to specified sentence length parameters
    filtered = [[], []]
    filtered_sents =  list()
    filtered_lens = list()

    # Sent filtering method to truncation by default (in case of anomalous input)
    if opt.sent_select == 'drop' or opt.sent_select == 'truncate' or opt.sent_select is None:
        sent_select = opt.sent_select
    else:
        sent_select = 'truncate'

    # Set filtering size to mean_len + (max_len - mean_len) // 2 by default
    observed_max_len = max(sent_lens)
    if opt.max_sent_len:
        target_len = opt.max_sent_len
    elif opt.sent_select is None:
        target_len = observed_max_len
    else:
        observed_mean_len = int(np.round(np.mean(sent_lens)))
        target_len = observed_mean_len + (observed_max_len - observed_mean_len) // 2

    for i in range(len(sim_data[0])):
        pair = sim_data[0][i]  # for each i is a question or a kb triple
        if len(pair[0].split()) > target_len or len(pair[1].split()) > target_len:
            if sent_select == 'drop':
                continue
            elif sent_select is None:
                pass
            else:
                pair_0 = ' '.join(pair[0].split()[:target_len])
                pair_1 = ' '.join(pair[1].split()[:target_len])
                if is_train:
                    pair_2 = ' '.join(pair[2].split()[:target_len])
                    pair = (pair_0, pair_1, pair_2)
                else:
                    pair = (pair_0, pair_1)

        filtered[0].append(pair)
        filtered[1].append(sim_data[1][i])
        if is_train:
            filtered_sents += [pair[0], pair[1], pair[2]]
            filtered_lens.append((len(pair[0]), len(pair[1]), len(pair[2])))
        else:
            filtered_sents += [pair[0], pair[1]]
            filtered_lens.append((len(pair[0]), len(pair[1])))

    # Generate corpus index dictionary and a list of pre-processed
    sim_vocab = Indexer(corpus_name)
    sim_vocab.set_target_len(target_len)

    print('Assembling dictionary ...')
    for i in range(len(filtered_sents)):
        sim_vocab.add_sentence(filtered_sents[i])  # if no external dictionary
    print('Registered %d unique words for the %s corpus.\n' % (sim_vocab.n_words, sim_vocab.name))
    return sim_vocab, filtered
