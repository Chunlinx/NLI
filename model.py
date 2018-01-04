import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.contrib.metrics import streaming_pearson_correlation
from helper import last_relevant_output

class model(object):
    def __init__(
            self, max_sequence_length,
            total_classes,
            embedding_size,
            id2Vecs,
            batch_size,
            lmd=1e-4
    ):
        # placeholders
        self.sent1 = tf.placeholder(tf.int32, [batch_size, max_sequence_length], name="sent1")
        self.sent1_length = tf.placeholder(tf.int32, [batch_size], name="sent1_length")
        self.sent2 = tf.placeholder(tf.int32, [batch_size, max_sequence_length], name="sent2")
        self.sent2_length = tf.placeholder(tf.int32, [batch_size], name="sent2_length")

        self.embedding_size = embedding_size
        self.max_sequence_length  =max_sequence_length
        ## labels
        self.labels = tf.placeholder(tf.int32, [batch_size,total_classes], name="labels")
        ## dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")
        ## hidden units
        self.hidden_Units = 50
        self.total_classes = total_classes
        self.batch_size = batch_size
        self.id2Vecs = id2Vecs
        self.embedding_size = embedding_size
        self.sent_dim = 4096
        self.l2_loss = tf.constant(value=0.0, dtype=tf.float32)
        with tf.variable_scope('this-scope') as scope:
            self.right_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)
            self.left_lstm_cell = rnn.BasicLSTMCell(num_units=self.hidden_Units)

            self.right_lstm_cell = rnn.DropoutWrapper(self.right_lstm_cell, output_keep_prob=self.dropout_keep_prob)
            self.left_lstm_cell = rnn.DropoutWrapper(self.left_lstm_cell, output_keep_prob=self.dropout_keep_prob)
            sent_1 = self.get_word_emb(self.sent1, name="sent_1")
            scope.reuse_variables()
            sent_2 = self.get_word_emb(self.sent2, name="sent_2")
        (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,
                                                                  cell_bw=self.right_lstm_cell,
                                                                  inputs=sent_1,
                                                                  sequence_length=self.sent1_length,
                                                                  dtype=tf.float32)
        (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.left_lstm_cell,
                                                                  cell_bw=self.right_lstm_cell,
                                                                  inputs=sent_2,
                                                                  sequence_length=self.sent2_length,
                                                                  dtype=tf.float32)

        # last_fw_out_1 = last_relevant_output(fw_out_1, self.sent1_length)
        # last_bw_out_1 = last_relevant_output(bw_out_1, self.sent1_length)
        #
        # last_fw_out_2 = last_relevant_output(fw_out_2, self.sent2_length)
        # last_bw_out_2 = last_relevant_output(bw_out_2, self.sent2_length)

        # sent_1_repr = tf.layers.dropout(tf.concat([last_fw_out_1,last_bw_out_1], 1),rate=self.dropout)
        # sent_2_repr = tf.layers.dropout(tf.concat([last_fw_out_2,last_bw_out_2],1),rate=self.dropout)
        # sent_1_repr = tf.concat([last_fw_out_1, last_bw_out_1], 1)
        # sent_2_repr = tf.concat([last_fw_out_2, last_bw_out_2], 1)
        combined_output_1 = tf.concat([fw_out_1,bw_out_1], axis=2)
        self.out1 = tf.reshape(combined_output_1, shape=[self.batch_size, max_sequence_length * self.hidden_Units * 2])

        self.lstm_W = tf.get_variable(
                "lstm_W",
                shape=[2 * self.hidden_Units * self.max_sequence_length, self.sent_dim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
        out1 = tf.matmul(self.out1,self.lstm_W)
        combined_output_2 = tf.concat([fw_out_2, bw_out_2], axis=2)
        out2 = tf.reshape(combined_output_2, shape=[self.batch_size, max_sequence_length * self.hidden_Units * 2])
        out2 = tf.matmul(out2,self.lstm_W)


        self.dot_ = tf.layers.dropout(tf.multiply(out1,out2),rate=1.0 - self.dropout_keep_prob)
        # self.diff_ = tf.layers.dropout(tf.abs(tf.subtract(out1,out2)),rate=1.0 - self.dropout_keep_prob)
        # self.exp_diff = tf.layers.dropout(tf.exp(-tf.abs(tf.subtract(out1,out2))),rate= 1.0 - self.dropout_keep_prob)
        self.out1_ = tf.layers.dropout(out1,rate=1.0 - self.dropout_keep_prob)
        self.out2_ = tf.layers.dropout(out2,rate=1.0 - self.dropout_keep_prob)

        with tf.name_scope("last-layer"):
            self.W_f_1 = tf.get_variable(
                "W_f_1",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_2 = tf.get_variable(
                "W_f_2",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W_f_3 = tf.get_variable(
                "W_f_3",
                shape=[self.sent_dim, self.total_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            bias = tf.Variable(tf.constant(value=0.01, shape=[self.total_classes], name="bias"))

            final_score = tf.matmul(self.out1_,self.W_f_1) + tf.matmul(self.out2_,self.W_f_2) + tf.matmul(self.dot_,self.W_f_3) + bias
            self.final_score = tf.sigmoid(final_score)
            self.pred = tf.argmax(self.final_score, 1, name="pred")

        with tf.name_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.final_score, self.labels)
            self.loss += lmd * self.l2_loss

        with tf.name_scope("accuracy"):
            accuracy = tf.equal(self.pred, tf.argmax(self.labels, 1))
            self.acc = tf.reduce_mean(tf.cast(accuracy,"float"),name="accuracy")



    def bi_attention(self,x,y):
        # matrx = tf.matmul(x,y,transpose_b=True)
        print("attention")
        x = tf.reshape(x,[self.batch_size,self.max_sequence_length,2*self.hidden_Units])
        y = tf.reshape(y, [self.batch_size, self.max_sequence_length, 2 * self.hidden_Units])

        W1 = tf.get_variable("W1",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable("W3",shape=[2*self.hidden_Units,1],initializer=tf.contrib.layers.xavier_initializer())

        out_ = []
        for l in range(self.batch_size):
            # S = tf.zeros(shape=[self.max_sequence_length, self.max_sequence_length])
            S = []
            for i in range(self.max_sequence_length):
                s = []
                for j in range(self.max_sequence_length):
                    s.append( tf.matmul(tf.reshape(x[l,i,:],[1,2*self.hidden_Units]),W1)
                              + tf.matmul(tf.reshape(y[l,j,:],[1,2*self.hidden_Units]),W2)
                              +  tf.matmul(tf.reshape(tf.multiply(x[l,i,:],y[l,j,:]), [1, 2 * self.hidden_Units]), W3)
                              )
                S.append(s)
            S = tf.squeeze(tf.convert_to_tensor(S))
            a = tf.nn.softmax(S,dim=1)
            o1 = []
            for i in range(self.max_sequence_length):
                o2 = []
                for j in range(self.max_sequence_length):
                    o2.append(a[i,j]*x[l,j,:])
                o1.append(o2)
            print("batch : "+ str(l))
            out_.append(o1)

        out_ = tf.reduce_sum(tf.convert_to_tensor(out_),axis=2)
        return out_

    def get_word_emb(self, x, name):
        """
        :param x:
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope("word-embedding-layer"):
                self.embeddings = tf.Variable(initial_value=self.id2Vecs, dtype=tf.float32, name='embedding_lookup',trainable=False)
                word_embeddings = tf.nn.embedding_lookup(self.embeddings, x, name=name)

        return word_embeddings

    def get_out(self, sentences,sent_length):
        (fw_out,bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.left_lstm_cell, self.right_lstm_cell, sentences,sequence_length=sent_length,
                                                                   dtype=tf.float32)
        fw_out_last = last_relevant_output(fw_out, sent_length)
        bw_out_last = last_relevant_output(bw_out,sent_length)
        combined_output = tf.concat([fw_out_last,bw_out_last], axis=1)
        # out = tf.reshape(combined_output, shape=[self.batch_size, self.max_sequence_length, self.hidden_Units * 2])
        # out = combined_output[:,-1,:]
        return combined_output


        # convert data to slices

        # Initial state of the LSTM memory.
        # hidden_state = tf.zeros([self.batch_size,])
        #
        # self.sequence = tf.split(self.sentences,num_or_size_splits=max_sequence_length,axis=1)


# m = model(max_sequence_length=11,total_classes=4,embedding_size=300,id2Vecs=np.zeros([7,300]),batch_size=3)