import os
import pickle

from similarity_estimator.networks import AnswerSelection
from similarity_estimator.options import TestingOptions
from similarity_estimator.sim_util import load_similarity_data
from utils.data_server import DataServer
from utils.init_and_storage import load_network, add_pretrained_embeddings
import torch


opt = TestingOptions()
test_path = '../2016train_word_segmentation.txt'
_, corpus_data = load_similarity_data(opt, test_path, 'qa_test', is_train=False)

# Load pretrained parameters
pretrained_path = os.path.join(opt.save_dir, 'pretraining/pretrained.pkl')
with open(pretrained_path, 'rb') as f:
    pretrained_embeddings, pretrained_vocab = pickle.load(f)

# Initialize the similarity classifier, original vocab size =
if torch.cuda.is_available():
    classifier = AnswerSelection(pretrained_vocab.n_words, opt, is_train=False).cuda()
else:
    classifier = AnswerSelection(pretrained_vocab.n_words, opt, is_train=False)

load_network(classifier.encoder_a, 'encoder_question', '36', opt.pretraining_dir)
load_network(classifier.encoder_b, 'encoder_candidates', '36', opt.pretraining_dir)

# 测试 bs=1
test_loader = DataServer(corpus_data, pretrained_vocab, opt, is_train=False, use_buckets=False, volatile=True)

# performance
total_classification_divergence = 0.0
total_classification_loss = 0.0

former_question = ' '
candidate = {}
right_num = 0
question_num = 0
mrr = 0
location = -1

for i, data in enumerate(test_loader):
    if i >= opt.num_test_samples:
        average_classification_divergence = total_classification_divergence / opt.num_test_samples
        average_classification_loss = total_classification_loss / opt.num_test_samples
        accuracy = right_num / question_num
        mrr /= question_num
        print('right number = %d' % right_num)
        print('question number = %d' % question_num)
        print('Accuracy after examining %d samples = %.4f' % (opt.num_test_samples, accuracy))
        print('MRR after examining %d questions = %.4f' % (question_num, mrr))
        # print('=================================================\n'
        #       '= Testing concluded after examining %d samples. =\n'
        #       '= Average classification divergence is %.4f.  =\n'
        #       '= Average classification loss (MSE) is %.4f.  =\n'
        #       '=================================================' %
        #       (opt.num_test_samples, average_classification_divergence, average_classification_loss))
        break

    s1_var, s2_var, label_var = data

    sentence_a = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s1_var.data.numpy().tolist()])
    sentence_b = ' '.join([pretrained_vocab.index_to_word[int(idx[0])] if idx[0] != 0 else '' for idx in
                           s2_var.data.numpy().tolist()])

    classifier.qa_step(s1_var, s2_var, label_var)
    prediction = classifier.get_score()

    if sentence_a != former_question:  # start a new group
        question_num += 1
        if i != 0:
            # inverse = [(value, key) for key, value in candidate.items()]
            rank = sorted(candidate.items(), key=lambda d: d[1][0], reverse=False)  # 应该选择距离最小者
            # predict_right_pair = max(inverse)
            # predict_right_answer = predict_right_pair[1]
            if rank[0][1][1] == 1:
                # if candidate[predict_right_answer][1] == 1:
                right_num += 1
                print(former_question + '【' + rank[0][0].strip() + '】')
                # print(predict_right_answer)
            for loc in range(len(rank)):
                if rank[loc][1][1] == 1:
                    location = loc + 1
                    break
            print('the location of the right answer of question [%d] is [%d]' % (question_num, location))

            mrr += float(1/location)
        candidate = {}
        former_question = sentence_a
    if former_question == sentence_a:
        candidate[sentence_b] = (prediction.data[0].cpu().numpy(), label_var.data[0][0])

    # loss = classifier.loss.data[0]

    # divergence = abs((prediction - label_var).data[0])
    # total_classification_divergence += divergence
    # total_classification_loss += loss

    # if i % 200 == 0:
    print('Sample: %d\n'
              'Question: %s\n'
              'Candidate triple: %s\n'
              'Prediction: %.8f\n'
              'Ground truth: %.4f\n'
              # 'Accuracy: %.4f\n'
              # 'Loss: %.4f\n'
              %
              (i, sentence_a, sentence_b, prediction.data[0], label_var.data[0][0]))
