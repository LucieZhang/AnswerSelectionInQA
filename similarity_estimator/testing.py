import os
import pickle

from similarity_estimator.networks import AnswerSelection
from similarity_estimator.options import TestingOptions
from similarity_estimator.sim_util import load_similarity_data
from utils.data_server import DataServer
from utils.init_and_storage import load_network
import torch


opt = TestingOptions()
test_path = '../2016train_word_segmentation.txt'
# test_char_path = '../../../Data/2016test_char_seg.txt'
# test_path = 'E:/Lucy/Dissertation/code/ChineseKBQA/Data/NLPCC2017-OpenDomainQA/data/dbqatest_word_seg.txt'
# train_path = '../2016train_word_seg_rank.txt'
_, corpus_data = load_similarity_data(opt, test_path, 'qa_test', is_train=False, is_dbqa=False)
# _, pairwise_corpus_data = load_similarity_data(opt, train_path, 'training data', is_train=True)

# Load pretrained parameters
pretrained_path = os.path.join(opt.save_dir, 'pretraining/pretrained.pkl')
with open(pretrained_path, 'rb') as f:
    pretrained_embeddings, pretrained_vocab = pickle.load(f)
pretrained_char_path = os.path.join(opt.save_dir, 'pretraining/pretrained_char.pkl')
with open(pretrained_char_path, 'rb') as fc:
    pretrained_embeddings_char, vocab_char = pickle.load(fc)

if torch.cuda.is_available():
    selector = AnswerSelection(pretrained_vocab.n_words, vocab_char.n_words, opt, is_train=False).cuda()
else:
    selector = AnswerSelection(pretrained_vocab.n_words, vocab_char.n_words, opt, is_train=False)

# params = selector.encoder.state_dict()
# print('the params before restored:')
# print(params)

load_network(selector.encoder, 'encoder', 'latest', opt.pretraining_dir)

# test bs=1
test_loader = DataServer(corpus_data, pretrained_vocab, vocab_char, opt, is_train=False, use_buckets=False, volatile=True)
# pairwise_test_loader = DataServer(pairwise_corpus_data, pretrained_vocab, opt, is_train=True, use_buckets=False, volatile=True)
print('target len of vocab2 = %d' % vocab_char.target_len)
former_question = ' '
candidate = {}
right_num = 0
question_num = 0
mrr = 0
avgP = 0
location = -1


def accuracy(dista, distb, labels):
    margin = 0
    pred = (distb - dista - margin).cpu().data
    data_labels = labels.cpu().data
    count_sum = torch.mul(data_labels, pred).numpy()
    # count=[1 for]
    count = 0
    for item in count_sum:
        if item > 0:
            count += 1
    return count * 1.0 / dista.size()[0]
#
# accs = []
#
# for i, data in enumerate(pairwise_test_loader):
#
#     s1_var, s2_var, s3_var, label_var = data
#
#     sentence_a = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
#                            s1_var.data.numpy().tolist()])
#     sentence_b = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
#                            s2_var.data.numpy().tolist()])
#     sentence_c = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
#                            s3_var.data.numpy().tolist()])
#
#     pred1, pred2 = selector.test_step(s1_var, s2_var, s3_var, label_var)
#
#     acc = accuracy(pred1, pred2, label_var)
#     accs.append(acc * s1_var.size(1))  # right numbers in a batch
#
#     print('Sample: %d\n'
#               'Question: %s\n'
#               'Candidate1: %s | Candidate2: %s\n'
#               'Prediction1: %.8f | Prediction2: %.8f\n'
#               'Ground truth: %.4f\n'
#               'Loss: %.4f\n'
#               %
#               (i, sentence_a, sentence_b, sentence_c, pred1.data[0], pred2.data[0], label_var.data[0][0], selector.loss.data[0]))
#
# final_acc = sum(accs) / (len(accs) * opt.train_batch_size)
# print('accuracy = %.4f' % final_acc)


for i, data in enumerate(test_loader):

    s1_var, s2_var, c1_var, c2_var, label_var = data

    sentence_a = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s1_var.data.numpy().tolist()])
    sentence_b = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s2_var.data.numpy().tolist()])

    prediction = selector.qa_step(s1_var, s2_var, c1_var, c2_var, label_var)

    sentence_a = str(sentence_a).strip()
    sentence_b = str(sentence_b).strip()

    if sentence_a != former_question:  # start a new group
        question_num += 1
        # print('sentence_a: %s\n former_question: %s' % (sentence_a, former_question))
        if i != 0:
            # inverse = [(value, key) for key, value in candidate.items()]
            right_ans_num = 0  # the number of right answers of one question
            ap = 0  # to calculate MAP
            rank = sorted(candidate.items(), key=lambda d: d[1][0], reverse=False)  # 应该选择距离最小
            # predict_right_pair = max(inverse)
            # predict_right_answer = predict_right_pair[1]
            if rank[0][1][1] == 1:
                # if candidate[predict_right_answer][1] == 1:
                right_num += 1
                # print(former_question + '【' + rank[0][0].strip() + '】')
                # print(predict_right_answer)

            # calculate MRR
            for loc in range(len(rank)):
                if rank[loc][1][1] == 1:
                    location = loc + 1
                    break
            print('the location of the first right answer of question [%d] is [%d]' % (question_num, location))
            mrr += float(1 / location)

            # calculate MAP
            for loc in range(len(rank)):
                if rank[loc][1][1] == 1:
                    right_ans_num += 1
                    ap += right_ans_num / (loc + 1)
            if right_ans_num != 0:
                avgP += ap / right_ans_num

        candidate = {}
        former_question = sentence_a
    if former_question == sentence_a:
        candidate[sentence_b] = (prediction.data[0].cpu().numpy(), label_var.data[0][0])

    if i % 200 == 0:
        print('Sample: %d\n'
              'Question: %s\n'
              'Candidate triple: %s\n'
              'Prediction: %.8f\n'
              'Ground truth: %.4f\n'
              # 'Accuracy: %.4f\n'
              # 'Loss: %.4f\n'
              %
              (i, sentence_a, sentence_b, prediction.data[0], label_var.data[0][0]))


accuracy = right_num / question_num
mrr /= question_num
mAvgP = avgP / question_num
print('right number = %d' % right_num)
print('question number = %d' % question_num)
print('Accuracy after examining %d samples = %.4f' % (opt.num_test_samples, accuracy))
print('MRR after examining %d questions = %.4f' % (question_num, mrr))
print('MAP after examining %d questions = %.4f' % (question_num, mAvgP))
print('Average candidates number of a question is %d' % (float(opt.num_test_samples) / question_num))
