# from https://github.com/jind11/HSLN-Joint-Sentence-Classification.git

import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix


from .data_utils import minibatches, pad_sequences, get_chunks, PAD
from .general_utils import Progbar
from .base_model import BaseModel


class HANNModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(HANNModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.config.l2_reg_lambda)
        self.config = config


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size)
        self.document_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="document_lengths")

        # shape = (batch size, max length of documents in batch (how many sentences in one abstract), max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_ids")

        # shape = (batch_size, max_length of sentence)
        self.sentence_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of documents, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None, pad_tok=0):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = [], []
            for abstract in words:
                char_ids_abstract, word_ids_abstract = [], []
                for sent in abstract:
                    char_id, word_id = zip(*sent)
                    char_ids_abstract += [list(char_id)]
                    word_ids_abstract += [list(word_id)]
                char_ids += [char_ids_abstract]
                word_ids += [word_ids_abstract]
            _, document_lengths = pad_sequences(word_ids, pad_tok=pad_tok, nlevels=1)
            word_ids, sentence_lengths = pad_sequences(word_ids, pad_tok=pad_tok, nlevels=2)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=pad_tok, nlevels=3)
        else:
            _, document_lengths = pad_sequences(words, pad_tok=pad_tok, nlevels=1)
            word_ids, sentence_lengths = pad_sequences(words, pad_tok=pad_tok, nlevels=2)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.document_lengths: document_lengths,
            self.sentence_lengths: sentence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, nlevels=1)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, document_lengths


    def add_word_embeddings_op(self, word_ids, word_lengths, char_ids, dropout):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words", reuse=tf.AUTO_REUSE):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                    word_ids, name="word_embeddings")

        if self.config.use_chars:
            with tf.variable_scope("chars", reuse=tf.AUTO_REUSE):
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1]*s[2], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(word_lengths, shape=[s[0]*s[1]*s[2]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], s[2], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        word_embeddings = tf.nn.dropout(word_embeddings, dropout)

        return word_embeddings


    def add_logits_op(self, word_embeddings, sentence_lengths, document_lengths, dropout):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        s = tf.shape(word_embeddings)
        
        if self.config.use_chars:
            word_embeddings_dim = self.config.dim_word + 2 * self.config.hidden_size_char
        else:
            word_embeddings_dim = self.config.dim_word

        sentence_lengths = tf.reshape(sentence_lengths, shape=[s[0]*s[1]])
        
        if self.config.use_cnn:
            word_embeddings = tf.reshape(word_embeddings, 
                            shape=[s[0]*s[1], s[-2], word_embeddings_dim, 1])

            if self.config.use_attention:
                with tf.variable_scope("conv-attention", reuse=tf.AUTO_REUSE):
                    W_word = tf.get_variable("weight", dtype=tf.float32, 
                            initializer=self.initializer, regularizer=self.regularizer,
                            shape=[self.config.cnn_filter_num, self.config.attention_size])
                    b_word = tf.get_variable("bias", shape=[self.config.attention_size],
                            dtype=tf.float32, initializer=tf.zeros_initializer())
                    U_word = tf.get_variable("U-noreg", dtype=tf.float32, 
                            initializer=self.initializer, 
                            shape=[self.config.attention_size, self.config.attention_hop])

            if self.config.use_cnn_rnn:
                with tf.variable_scope("cnn-rnn"):
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)

            pooled_outputs = []
            for i, size in enumerate(self.config.cnn_filter_sizes):
                with tf.variable_scope("conv-%d" % size, reuse=tf.AUTO_REUSE):# , reuse=False
                    W_conv = tf.get_variable(name='weight', initializer=self.initializer, 
                                            shape=[size, word_embeddings_dim, 1, self.config.cnn_filter_num], 
                                            regularizer=self.regularizer)
                    b_conv = tf.get_variable(name='bias', initializer=tf.zeros_initializer(), 
                                            shape=[self.config.cnn_filter_num])
                    conv = tf.nn.conv2d(word_embeddings, W_conv, strides=[1, 1, word_embeddings_dim, 1],
                                        padding="SAME")

                    h = tf.nn.tanh(tf.nn.bias_add(conv, b_conv), name="h") # bz, n, 1, dc
                    # h = tf.nn.max_pool(h,
                    #         ksize=[1, 2, 1, 1],
                    #         strides=[1, 2, 1, 1],
                    #         padding="SAME")
                    h = tf.squeeze(h, axis=2) # bz, n, dc

                    if self.config.use_cnn_rnn:
                        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                                cell_fw, cell_bw, h,
                                sequence_length=sentence_lengths, dtype=tf.float32)
                        pooled = tf.concat([output_fw, output_bw], axis=-1) # bz, dc
                    else:
                        if self.config.use_attention:
                            U_sent = tf.tanh(tf.matmul(tf.reshape(h, shape=[-1, self.config.cnn_filter_num]), 
                                                                W_word) + b_word) # (bz*len, attn_size)
                            A = tf.transpose(tf.reshape(tf.matmul(U_sent, U_word), shape=[-1, s[2], 
                                                    self.config.attention_hop]), perm=[0, 2, 1]) # (bz, attn_hop, len)
                            A += 100000. * (tf.tile(tf.expand_dims(tf.cast(tf.sequence_mask(sentence_lengths), tf.float32), axis=1), 
                                                            [1, self.config.attention_hop, 1]) - 1)
                            self.A = tf.nn.softmax(A) # (bz, attn_hop, len)
                            pooled = tf.reshape(tf.einsum('aij,ajk->aik', self.A, h), shape=[-1, 
                                                self.config.attention_hop*self.config.cnn_filter_num])
                        else:
                            # max pooling
                            pooled = tf.reduce_max(h, axis=1) # bz, dc
                    
                    pooled_outputs.append(pooled)

            output = tf.concat(pooled_outputs, axis=-1) 
            # dropout
            output = tf.nn.dropout(output, dropout)
            
            if self.config.use_cnn_rnn:
                cnn_filter_tot_num = (2 * self.config.hidden_size_lstm_sentence) * len(self.config.cnn_filter_sizes)
            else:
                cnn_filter_tot_num = len(self.config.cnn_filter_sizes) * self.config.cnn_filter_num * self.config.attention_hop

            if self.config.use_document_level == True:
                output = tf.reshape(output, 
                            shape=[-1, s[1], cnn_filter_tot_num])
        else:
            word_embeddings = tf.reshape(word_embeddings, 
                                shape=[s[0]*s[1], s[-2], word_embeddings_dim])

            if self.config.use_attention:
                with tf.variable_scope("bi-lstm-sentence", reuse=tf.AUTO_REUSE):
                    if self.config.use_gru:
                        cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm_sentence)
                        cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm_sentence)
                    else:
                        cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                        cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)

                    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, word_embeddings,
                            sequence_length=sentence_lengths, dtype=tf.float32)
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    W_word = tf.get_variable("weight", dtype=tf.float32, 
                            initializer=self.initializer, regularizer=self.regularizer,
                            shape=[2*self.config.hidden_size_lstm_sentence, self.config.attention_size])
                    b_word = tf.get_variable("bias", shape=[self.config.attention_size],
                            dtype=tf.float32, initializer=tf.zeros_initializer())
                    U_word = tf.get_variable("U-noreg", dtype=tf.float32, 
                            initializer=self.initializer, 
                            shape=[self.config.attention_size, self.config.attention_hop])

                    output = tf.reshape(output, shape=[-1, 2*self.config.hidden_size_lstm_sentence])
                    U_sent = tf.tanh(tf.matmul(output, W_word) + b_word) # (bz*len, attn_size)
                    A = tf.transpose(tf.reshape(tf.matmul(U_sent, U_word), shape=[-1, s[2], self.config.attention_hop]), perm=[0, 2, 1]) # (bz, attn_hop, len)
                    A += 100000. * (tf.tile(tf.expand_dims(tf.cast(tf.sequence_mask(sentence_lengths), tf.float32), axis=1), 
                                                    [1, self.config.attention_hop, 1]) - 1)
                    self.A = tf.nn.softmax(A) # (bz, attn_hop, len)
                    output = tf.reshape(output, shape=[-1, s[2], 2*self.config.hidden_size_lstm_sentence]) # (bz, len, hidden_size)
                    output = tf.reshape(tf.einsum('aij,ajk->aik', self.A, output), shape=[-1, 
                                        self.config.attention_hop*2*self.config.hidden_size_lstm_sentence])

            else:
                with tf.variable_scope("bi-lstm-sentence", reuse=tf.AUTO_REUSE):
                    if self.config.use_gru:
                        cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm_sentence)
                        cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm_sentence)
                    else:
                        cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                        cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_sentence)
                    _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw, cell_bw, word_embeddings,
                            sequence_length=sentence_lengths, dtype=tf.float32)
                    output = tf.concat([output_fw, output_bw], axis=-1)

            # dropout
            output = tf.nn.dropout(output, dropout)

            if self.config.use_document_level == True:
                output = tf.reshape(output, [-1, s[1], self.config.attention_hop*2*self.config.hidden_size_lstm_sentence])

        if self.config.use_document_level == True:
            with tf.variable_scope("bi-lstm-document", reuse=tf.AUTO_REUSE):
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_document)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm_document)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, output,
                    sequence_length=document_lengths, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                # dropout
                output = tf.nn.dropout(output, dropout)
                output = tf.reshape(output, shape=[-1, 2*self.config.hidden_size_lstm_document])

        if self.config.use_document_level == True:
            hidden_size = 2 * self.config.hidden_size_lstm_document
        else:
            if self.config.use_cnn:
                hidden_size = cnn_filter_tot_num
            else:
                hidden_size = self.config.attention_hop * 2 * self.config.hidden_size_lstm_sentence

        with tf.variable_scope("proj", reuse=tf.AUTO_REUSE):
            W_infer = tf.get_variable("weight", dtype=tf.float32, 
                    initializer=self.initializer, regularizer=self.regularizer,
                    shape=[hidden_size, self.config.ntags])

            b_infer = tf.get_variable("bias", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W_infer) + b_infer
            logits = tf.reshape(pred, [-1, s[1], self.config.ntags])

        return logits


    def forward(self, word_ids, char_ids, word_lengths, sentence_lengths, document_lengths, dropout):
        word_embeddings = self.add_word_embeddings_op(word_ids, word_lengths, char_ids, dropout)
        logits = self.add_logits_op(word_embeddings, sentence_lengths, document_lengths, dropout)

        return logits

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)

    def Frobenius(self, tensor):
        # print(tf.rank(tensor), tf.shape(tensor))
        # if tf.rank(tensor) == 3:  # batched matrix
        return tf.reduce_mean((tf.squeeze(tf.reduce_sum(tensor**2, [1, 2])) + 1e-10) ** 0.5)
        # else:
            # raise Exception('matrix for computing Frobenius norm should be with 3 dims')


    def add_loss_op(self, logits, logits_no_dropout, labels, document_lengths):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    logits, labels, document_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)
            mask = tf.sequence_mask(document_lengths)
            losses = tf.boolean_mask(losses, mask)
            loss = tf.reduce_mean(losses)

        # add l2 regularization
        l2 = self.config.l2_reg_lambda * sum([
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name)])
        loss += l2

        # add dropout loss
        if logits_no_dropout is not None:
            self.drop_loss = tf.nn.l2_loss(tf.subtract(logits, logits_no_dropout))
            loss += self.config.drop_penalty * self.drop_loss

        # add attention matrix penalty
        if self.config.attention_hop > 1:
            A_T = tf.transpose(self.A, perm=[0, 2, 1])
            self.attention_loss = self.Frobenius(tf.einsum('aij,ajk->aik', self.A, A_T) - \
                tf.eye(self.config.attention_hop, batch_shape=[tf.shape(self.A)[0]]))
            loss += self.config.attention_penalty * self.attention_loss

        # for tensorboard
        tf.summary.scalar("loss", loss)

        return loss


    def build(self):
        # NER specific functions
        tf.reset_default_graph()
        self.add_placeholders()
        self.logits = self.forward(self.word_ids, self.char_ids, self.word_lengths, 
                self.sentence_lengths, self.document_lengths, self.dropout)

        if self.config.drop_penalty > 0.:
            self.logits_no_dropout = self.forward(self.word_ids, self.char_ids, self.word_lengths, 
                self.sentence_lengths, self.document_lengths, 1.0)
        else:
            self.logits_no_dropout = None

        self.add_pred_op()
        self.loss = self.add_loss_op(self.logits, self.logits_no_dropout, 
                                    self.labels, self.document_lengths)

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                        self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            document_length

        """

        fd, document_lengths = self.get_feed_dict(words, dropout=1.0)
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # print(trans_params)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, document_length in zip(logits, document_lengths):
                logit = logit[:document_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, document_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, document_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            if self.config.drop_penalty > 0 and self.config.attention_hop > 1:
                _, train_loss, summary, attention_loss, drop_loss = self.sess.run(
                    [self.train_op, self.loss, self.merged, 
                    self.attention_loss, self.drop_loss], feed_dict=fd)
            elif self.config.drop_penalty > 0:
                _, train_loss, summary, drop_loss = self.sess.run(
                    [self.train_op, self.loss, self.merged, 
                    self.drop_loss], feed_dict=fd)
            elif self.config.attention_hop > 1:
                _, train_loss, summary, attention_loss = self.sess.run(
                    [self.train_op, self.loss, self.merged, 
                    self.attention_loss], feed_dict=fd)
            else:
                _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            if not self.config.train_accuracy:
                prog.update(i + 1, [("train loss", train_loss)])
            else:
                labels_pred, document_lengths = self.predict_batch(words)
                accs = []
                for lab, lab_pred, length in zip(labels, labels_pred,
                                                 document_lengths):
                    lab      = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs    += [a==b for (a, b) in zip(lab, lab_pred)]
                acc = np.mean(accs)
                if self.config.drop_penalty > 0 and self.config.attention_hop > 1:
                    prog.update(i + 1, [("train loss", train_loss), ("attentin loss", attention_loss),
                        ("drop loss", drop_loss), ("accuracy", acc)])
                elif self.config.attention_hop > 1:
                    prog.update(i + 1, [("train loss", train_loss), ("attentin loss", attention_loss),
                         ("accuracy", acc)])
                elif self.config.drop_penalty > 0:
                    prog.update(i + 1, [("train loss", train_loss), ("drop loss", drop_loss), 
                        ("accuracy", acc)])
                else:
                    prog.update(i + 1, [("train loss", train_loss), ("accuracy", acc)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items() if not('report' in k or 'matrix' in k)])
        self.logger.info(msg)

        return metrics["weighted-f1"]


    def run_evaluate(self, test, report=True):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        labs = []
        labs_pred = []
        
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, document_lengths = self.predict_batch(words)
            for lab, lab_pred, length in zip(labels, labels_pred,
                                             document_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                labs.extend(lab)
                labs_pred.extend(lab_pred)

        labs = [self.idx_to_tag[lab].split('_')[0] for lab in labs]
        labs_pred = [self.idx_to_tag[lab_pred].split('_')[0] for lab_pred in labs_pred]
        _, _, macro_f1, _ = precision_recall_fscore_support(labs, labs_pred, average='macro')
        _, _, micro_f1, _ = precision_recall_fscore_support(labs, labs_pred, average='micro')
        _, _, weighted_f1, _ = precision_recall_fscore_support(labs, labs_pred, average='weighted')
        acc = np.mean(accs)

        if report == True:
            class_report = classification_report(labs, labs_pred, digits=4)
            print(class_report)
            confusion = confusion_matrix(labs, labs_pred)
            print(confusion)

        return {"acc": 100*acc, "macro-f1": 100*macro_f1, "micro-f1": 100*micro_f1, 
                "weighted-f1": 100*weighted_f1, "classification-report": class_report, 
                "confusion-matrix": confusion}
    
    