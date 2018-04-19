import os
import pickle
import time

import numpy as np
import torch
from utils.data_server import DataServer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from similarity_estimator.networks import AnswerSelection
from similarity_estimator.options import TestingOptions
# from similarity_estimator.sick_extender import SickExtender
from similarity_estimator.sim_util import load_similarity_data
from utils.init_and_storage import add_pretrained_embeddings, extend_embeddings, update_learning_rate, save_network
from utils.parameter_initialization import xavier_normal

from similarity_estimator.laplotter import LossAccPlotter


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


opt = TestingOptions()

if opt.pre_training:
    save_dir = opt.pretraining_dir
    # sts_corpus_path = os.path.join(opt.data_dir, '')
    sts_corpus_path = '../2016train_word_seg_rank.txt'
    # pretrained_embed_path = './news12g_bdbk20g_nov90g_dim128.model'
    vocab, corpus_data = load_similarity_data(opt, sts_corpus_path, 'QA_train_corpus')
    init_embeddings = xavier_normal(torch.randn([vocab.n_words, 128])).numpy()
    # Add FastText embeddings
    external_embeddings = add_pretrained_embeddings(
        init_embeddings, vocab, os.path.join(opt.data_dir, 'news12g_bdbk20g_nov90g_dim128.bin'))

    if torch.cuda.is_available():
        classifier = AnswerSelection(vocab.n_words, opt, pretrained_embeddings=external_embeddings, is_train=True).cuda()
    else:
        classifier = AnswerSelection(vocab.n_words, opt, pretrained_embeddings=external_embeddings, is_train=True)

    classifier.initialize_parameters()
    # load the pre-trained embedding table
    classifier.encoder_a.embedding_table.weight.data.copy_(external_embeddings)
    classifier.encoder_b.embedding_table.weight.data.copy_(external_embeddings)
    classifier.encoder_c.embedding_table.weight.data.copy_(external_embeddings)

learning_rate = opt.learning_rate

best_validation_accuracy = 0
epochs_without_improvement = 0
final_epoch = 0

train_data, valid_data, train_labels, valid_labels = train_test_split(corpus_data[0], corpus_data[1],
                                                                      test_size=0.1, random_state=0)
plotter = LossAccPlotter(title='Learning Curve', save_to_filepath='./img/learning_curve-att3.png',
                         show_regressions=False,
                         show_averages=False, show_loss_plot=True, show_acc_plot=True, x_label='Epoch')

for epoch in range(opt.num_epochs):
    epoch_start_time = time.time()
    running_loss = list()
    total_train_loss = list()  # for epoch

    train_loader = DataServer([train_data, train_labels], vocab, opt, shuffle=opt.shuffle, is_train=True,
                              use_buckets=True, volatile=False)

    accs = []

    for i, data in enumerate(train_loader):
        # running_loss = []  # for batch
        s1_var, s2_var, s3_var, label_var = data

        dista, distb = classifier.train_step(s1_var, s2_var, s3_var, label_var)  # , prediction_positive_qa, newGroup)
        acc = accuracy(dista, distb, label_var)
        accs.append(acc * s1_var.size(0))  # right numbers in a batch
        train_batch_loss = classifier.loss.data[0]  # batch_size * 1
        running_loss.append(train_batch_loss)
        total_train_loss.append(train_batch_loss)

        if i % opt.report_freq == 0 and i != 0:
            running_avg_loss = sum(running_loss) / len(running_loss)
            print('Epoch: %d | Training Batch: %d | Average loss since batch %d: %.4f | Average acc %.4f' %
                  (epoch, i, i - opt.report_freq, running_avg_loss, acc))  # sum(accs) / (len(accs) * s1_var.size(0))
            running_loss = list()

    # Epoch summarize
    avg_training_loss = sum(total_train_loss) / len(total_train_loss)
    avg_training_accuracy = sum(accs) / (len(accs) * s1_var.size(0))
    print('Average training batch loss at epoch %d: %.4f | Average batch accuracy: %.4f' %
          (epoch, avg_training_loss, avg_training_accuracy))

    print('time consumed %5.2f s' % (time.time() - epoch_start_time))

    # Validate each epoch
    valid_batch_loss = []
    if False:
        total_valid_loss = list()

        valid_loader = DataServer([valid_data, valid_labels], vocab, opt, is_train=True, use_buckets=True,
                                  volatile=True)

        # Validation
        accs = []
        for i, data in enumerate(valid_loader):
            s1_var, s2_var, s3_var, label_var = data
            distc, distd = classifier.test_step(s1_var, s2_var, s3_var, label_var)
            acc = accuracy(distc, distd, label_var)
            accs.append(acc * s1_var.size(0))
            valid_batch_loss = classifier.loss.data[0]
            total_valid_loss.append(valid_batch_loss)

            if i % opt.report_freq == 0 and i != 0:
                valid_batch_loss = sum(total_valid_loss) / len(total_valid_loss)
                print('Epoch: %d | Training Batch: %d | Average loss since batch %d: %.4f | Average acc %.4f' %
                      (epoch, i, i - opt.report_freq, valid_batch_loss, sum(accs) / (len(accs) * s1_var.size(0))))
                # valid_batch_loss = list()

        # Report fold statistics
        # avg_valid_accuracy = sum(total_valid_loss) / len(total_valid_loss)
        # print('Average validation fold accuracy at epoch %d: %.4f' % (epoch, avg_valid_accuracy))
        # Save network parameters if performance has improved
        # if avg_valid_accuracy >= best_validation_accuracy:  # original: <= is wrong
        #     epochs_without_improvement += 1
        # else:
        #     best_validation_accuracy = avg_valid_accuracy
        #     epochs_without_improvement = 0
        #     save_network(classifier.encoder_a, 'qa_classifier', 'latest', save_dir)

    # if epoch >= opt.start_early_stopping:
    #     loss_val = avg_valid_accuracy  # sum(total_valid_loss)
    #     acc_val = avg_valid_accuracy
    # else:
    #     loss_val, acc_val = None, None
    plotter.add_values(epoch, loss_train=avg_training_loss, acc_train=avg_training_accuracy
                       )

    plotter.block()

    # Save network parameters at the end of each nth epoch
    if epoch % opt.save_freq == 0 and epoch != 0:
        print('Saving model networks after completing epoch %d' % epoch)
        save_network(classifier.encoder_a, 'encoder_question', epoch, save_dir)
        save_network(classifier.encoder_b, 'encoder_candidates', epoch, save_dir)

    # Anneal learning rate:
    if epochs_without_improvement == opt.start_annealing:
        old_learning_rate = learning_rate
        learning_rate *= opt.annealing_factor
        update_learning_rate(classifier.optimizer_a, learning_rate)
        update_learning_rate(classifier.optimizer_b, learning_rate)
        update_learning_rate(classifier.optimizer_c, learning_rate)
        print('Learning rate has been updated from %.4f to %.4f' % (old_learning_rate, learning_rate))

    # Terminate training early, if no improvement has been observed for n epochs
    if epochs_without_improvement >= opt.patience:
        print('Stopping training early after %d epochs, following %d epochs without performance improvement.' %
              (epoch, epochs_without_improvement))
        final_epoch = epoch
        break

print('Training procedure concluded after %d epochs total. Best validated epoch: %d.'
      % (final_epoch, final_epoch - opt.patience))

if opt.pre_training:
    # Save pretrained embeddings and the vocab object
    pretrained_path = os.path.join(save_dir, 'pretrained.pkl')
    pretrained_embeddings = classifier.encoder_a.embedding_table.weight.data  # b is different with a, need modification
    with open(pretrained_path, 'wb') as f:
        pickle.dump((pretrained_embeddings, vocab), f)
    print('Pre-trained parameters saved to %s' % pretrained_path)

if not opt.pre_training:
    ''' Regression step over the training set to improve the predictive power of the model'''
    # Obtain similarity score predictions for each item within the training corpus
    labels = list()
    predictions = list()

    # Initiate the training data loader
    train_loader = DataServer([train_data, train_labels], vocab, opt, is_train=True, volatile=True)

    # Obtaining predictions
    for i, data in enumerate(train_loader):
        # Obtain data
        s1_var, s2_var, label_var = data
        labels += [l[0] for l in label_var.data.numpy().tolist()]
        classifier.test_step(s1_var, s2_var, label_var)
        batch_predict = classifier.prediction.data.squeeze().numpy().tolist()
        predictions += batch_predict

    labels = np.array(labels)
    predictions = np.array(predictions).reshape(-1, 1)

    # Fit an SVR (following the scikit-learn tutorial)
    sim_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                                                           "gamma": np.logspace(-2, 2, 5)})

    sim_svr.fit(predictions, labels)
    print('SVR complexity and bandwidth selected and model fitted successfully.')

    # Save trained SVR model
    svr_path = os.path.join(save_dir, 'sim_svr.pkl')
    with open(svr_path, 'wb') as f:
        pickle.dump(sim_svr, f)
    print('Trained SVR model saved to %s' % svr_path)
