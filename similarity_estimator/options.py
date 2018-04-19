import os


class TestingOptions(object):

    def __init__(self):
        # Data
        self.max_sent_len = None
        self.pad = True
        self.freq_bound = 1  # 3
        self.shuffle = False  # refer to DataLoader
        self.sent_select = 'truncate'
        self.lower = False
        self.num_buckets = 3

        # Network
        self.embedding_dims = 128
        self.hidden_dims = 50
        self.num_layers = 1
        self.train_batch_size = 20
        self.test_batch_size = 1
        self.clip_value = 0.25
        self.learning_rate = 0.001
        self.beta_1 = 0.5

        self.da = 350  # attention unit number
        self.r = 1  # num of parts want to focus on
        self.mlp_nhid = 3000
        self.nclass = 2
        self.pooling = 'all'
        self.dropout = 0.5

        self.pre_training = True
        self.num_epochs = 40  # 100

        self.start_early_stopping = 2  # 2
        self.patience = 20
        self.start_annealing = 4
        self.annealing_factor = 0.75

        # Training
        self.report_freq = 1
        self.save_freq = 4
        # self.home_dir = os.path.join(os.path.dirname(__file__), '..')
        self.home_dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.home_dir, 'data')
        # self.save_dir = os.path.join(self.home_dir, 'similarity_estimator/models')
        self.save_dir = os.path.join(self.home_dir, 'models')
        self.pretraining_dir = os.path.join(self.save_dir, 'pretraining')

        # Testing
        self.num_test_samples = 1577  # 76567

