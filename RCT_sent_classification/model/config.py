import os

from .general_utils import get_logger
from .data_utils import get_trimmed_wordvec_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, parser, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        ## parse args
        self.parser = parser
        # training parameters
        parser.add_argument('--nepochs', default='400', type=int,
                    help='number of epochs')
        parser.add_argument('--dropout', default='0.5', type=float,
                    help='number of epochs')
        parser.add_argument('--batch_size', default='10', type=int,
                    help='batch size')
        parser.add_argument('--lr', default='0.03', type=float,
                    help='learning rate')
        parser.add_argument('--lr_method', default='adam', type=str,
                    help='optimization method')
        parser.add_argument('--lr_decay', default='0.99', type=float,
                    help='learning rate decay rate')
        parser.add_argument('--clip', default='10', type=float,
                    help='gradient clipping')
        parser.add_argument('--nepoch_no_imprv', default='10', type=int,
                    help='number of epoch patience')
        parser.add_argument('--l2_reg_lambda', default='0.0000', type=float,
                    help='l2 regularization coefficient')
        parser.add_argument('--drop_penalty', default='0.01', type=float,
                    help='Penalty coefficient for dropout')
        parser.add_argument('--attention_penalty', default='0.0000', type=float,
                    help='Penalty coefficient for attention matrix')

        # data and results paths
        parser.add_argument('--dataset_name', default='study_arms', type=str,
                    help='the name of dataset to use')
        parser.add_argument('--dir_output', default='test', type=str,
                    help='directory for output')
        parser.add_argument('--filename_wordvec_trimmed', default='word2vec_pubmed.trimmed.txt', 
                    type=str, help='directory for trimmed word embeddings file')
        parser.add_argument('--filename_wordvec', default='PubMed-w2v.txt', 
                    type=str, help='directory for original word embeddings file')

        # model hyperparameters
        parser.add_argument('--hidden_size_char', default='100', type=int,
                    help='hidden size of character level lstm')
        parser.add_argument('--hidden_size_lstm_sentence', default='200', type=int,
                    help='hidden size of sentence level lstm')
        parser.add_argument('--hidden_size_lstm_document', default='200', type=int,
                    help='hidden size of document level lstm')
        parser.add_argument('--attention_size', default='200', type=int,
                    help='attention vector size')
        parser.add_argument('--attention_hop', default='15', type=int,
                    help='number of attention hops')
        parser.add_argument('--cnn_filter_num', default='200', type=int,
                    help='number of cnn filters for each window size')
        parser.add_argument('--dim_char', default='100', type=int,
                    help='character embedding dimension')
        parser.add_argument('--cnn_filter_sizes', default='2,3,4,5', type=str,
                    help='cnn filter window sizes')

        # misc
        parser.add_argument('--restore', action='store_true', 
                    help='whether restore from previous trained model')
        parser.add_argument('--use_crf',default=True, action='store_false', 
                    help='whether use crf optimization layer')
        parser.add_argument('--use_chars', action='store_true', 
                    help='whether use character embeddings')
        parser.add_argument('--use_document_level',default=1, action='store_false', 
                    help='whether use document level lstm layer')
        parser.add_argument('--use_attention', action='store_false', 
                    help='whether use attention based pooling')
        parser.add_argument('--use_cnn', action='store_true', 
                    help='whether use cnn or lstm for sentence representation')
        parser.add_argument('--train_embeddings', action='store_true', 
                    help='whether use cnn or lstm for sentence representation')
        parser.add_argument('--use_pretrained',default=True, action='store_false', 
                    help='whether use cnn or lstm for sentence representation')
        parser.add_argument('--train_accuracy', action='store_false', 
                    help='whether report accuracy while training')
        parser.add_argument('--use_cnn_rnn', action='store_true', 
                    help='whether stack rnn layer over cnn for sentence classification')
        parser.add_argument('--use_gru', action='store_true', 
                    help='whether gru instead of lstm')

        self.parser.parse_args(namespace=self)

        
        self.dataset_name = "study_arms"#news
        self.dir_output = "study_arms"#"news_class"
        
        
        
        self.filename_wordvec = os.path.join('word2vec', 
                                            self.filename_wordvec)
        self.dir_output = os.path.join('results', self.dir_output)
        self.dir_model  = os.path.join(self.dir_output, "model.weights")
        self.path_log   = os.path.join(self.dir_output, "log.txt")

        self.cnn_filter_sizes = [int(i) for i in self.cnn_filter_sizes.split(',')]
        self.filename_dev = os.path.join(self.dataset_name,"dev.txt")
        self.filename_test = os.path.join(self.dataset_name,"test.txt")
        self.filename_train = os.path.join(self.dataset_name,"train.txt")
        self.filename_words = os.path.join(self.dataset_name,"words.txt")
        self.filename_tags = os.path.join(self.dataset_name,"tags.txt")
        self.filename_chars = os.path.join(self.dataset_name,"chars.txt")
        self.filename_wordvec_trimmed = os.path.join(self.dataset_name, self.filename_wordvec_trimmed)
        '''
        if self.dataset_name == 'pubmed-20k':
            self.filename_dev = "data/PubMed_20k_RCT/dev_clean.txt"
            self.filename_test = "data/PubMed_20k_RCT/test_clean.txt"
            self.filename_train = "data/PubMed_20k_RCT/train_clean.txt"
            self.filename_words = "data/PubMed_20k_RCT/words.txt"
            self.filename_tags = "data/PubMed_20k_RCT/tags.txt"
            self.filename_chars = "data/PubMed_20k_RCT/chars.txt"
            self.filename_wordvec_trimmed = os.path.join('data/PubMed_20k_RCT', self.filename_wordvec_trimmed)
        elif self.dataset_name == 'pubmed-200k':
            self.filename_dev = "data/PubMed_200k_RCT/dev_clean.txt"
            self.filename_test = "data/PubMed_200k_RCT/test_clean.txt"
            self.filename_train = "data/PubMed_200k_RCT/train_clean.txt"
            self.filename_words = "data/PubMed_200k_RCT/words.txt"
            self.filename_tags = "data/PubMed_200k_RCT/tags.txt"
            self.filename_chars = "data/PubMed_200k_RCT/chars.txt"
            self.filename_wordvec_trimmed = os.path.join('data/PubMed_200k_RCT', self.filename_wordvec_trimmed)
        elif self.dataset_name == 'nicta':
            self.filename_dev = "data/nicta_piboso/dev_clean.txt"
            self.filename_test = "data/nicta_piboso/test_clean.txt"
            self.filename_train = "data/nicta_piboso/train_clean.txt"
            self.filename_words = "data/nicta_piboso/words.txt"
            self.filename_tags = "data/nicta_piboso/tags.txt"
            self.filename_chars = "data/nicta_piboso/chars.txt"
            self.filename_wordvec_trimmed = os.path.join('data/nicta_piboso', self.filename_wordvec_trimmed)
        else:
            raise 'No such dataset!'
        '''
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # log the attributes
        #msg = ', '.join(['{}: {}'.format(attr, getattr(self, attr)) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")])
        #self.logger.info(msg)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_wordvec_vectors(self.filename_wordvec_trimmed, self.vocab_words)
                if self.use_pretrained else None)
        self.dim_word = (self.embeddings.shape[1] if self.use_pretrained else 200)

    max_iter = None # if not None, max number of examples in Dataset


