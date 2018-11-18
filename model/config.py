import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word
from parser_config import Config as parser_Config


#=================== MODIFY!!!!!=================
#data_dir = "tian_iob/set6"
#results_dir = os.path.join(data_dir,"result_no_pre_1")
parser_config = parser_Config()
data_dir = parser_Config.data_dir
pretrain_dir = parser_Config.pretrain_dir
results_dir = parser_Config.pretrain_dir
# ============================================
class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

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
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

    cwd = os.getcwd()

    # general config
    dir_output = os.path.join(cwd, results_dir)
    dir_model  = os.path.join(dir_output, "model.weights/")
    path_log   = os.path.join(dir_output, "log.txt")
    dir_pretrain = os.path.join(pretrain_dir,"model.weights/")
    
    # embeddings
    dim_word = 200
    dim_char = 100

    # glove files
    # filename_glove = os.path.join(cwd, "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    #change to PubMed

    #filename_glove = os.path.join(cwd, "/Users/tk2624/projects/EBM-PICO/EBM-NLP-master/acl_scripts/lstm-crf/data/embedding/PubMed-w2v.txt")
    filename_glove = "/home/tk2624/projects/lstm-crf-py3/data/embeddings/PubMed-w2v.txt"

    # trimmed embeddings (created from glove_filename with build_data.py)
    #filename_trimmed = os.path.join(cwd, "data/embeddings.{}d.trimmed.npz".format(dim_word))
    filename_trimmed = os.path.join(cwd, data_dir,"embeddings.{}d.trimmed.npz".format(dim_word))
    use_pretrained = True
    data_dir = data_dir
    # dataset
    filename_dev = os.path.join(cwd, data_dir,"set2_dev.conll.clean")#"dev_span.iob.txt")
    filename_test = os.path.join(cwd, data_dir,"set2_test.conll.clean")#"gold_span.iob.txt")
    filename_train = os.path.join(cwd, data_dir,"set2_train.conll.clean")#"train_span.iob.txt")

    #filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = os.path.join(cwd, data_dir,"words.txt")
    filename_tags = os.path.join(cwd, data_dir,"tags.txt")
    filename_chars = os.path.join(cwd, data_dir,"chars.txt")

    # training
    train_embeddings = False
    nepochs          = 80
    dropout          = 0.5
    batch_size       = 5
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 200 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
