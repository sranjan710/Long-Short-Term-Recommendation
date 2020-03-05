from __future__ import division

import tensorflow as tf
import pickle
import numpy as np
tf.compat.v1.disable_resource_variables()
class Dis():
    def __init__(self, itm_cnt, usr_cnt, rnn_size,mf_emb_dim,fc_feat_size,image_emb_size,
                 input_emb_dim,n_time_step, mixture,use_cnn,mixture_score,
                 learning_rate, beta1, grad_clip, lamda=0.2, initdelta=0.05,MF_paras=None,
                 model_type="rnn",use_sparse_tensor=True, update_rule="sgd",pairwise=False):
        """
        Args:
            dim_itm_embed: (optional) Dimension of item embedding.
            dim_usr_embed: (optional) Dimension of user embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            usr_cnt: (optional) The size of all users.
            itm_cnt: (optional) The size of all items.
        """
        tf.compat.v1.set_random_seed(123)
        self.V_M = itm_cnt
        self.V_U = usr_cnt
        self.T = n_time_step
        self.rnn_size = rnn_size
        self.fc_feat_size = fc_feat_size
        self.image_emb_size = image_emb_size
        self.mf_emb_dim = mf_emb_dim
        self.input_emb_dim = input_emb_dim

        self.param_mf = MF_paras
        self.paras_rnn = []

        self.model_type = model_type
        self.mixture = mixture
        self.mixture_score = mixture_score

        self.user_sequence = tf.compat.v1.placeholder(tf.float32, [None, self.T, self.V_M])
        self.item_sequence = tf.compat.v1.placeholder(tf.float32, [None, self.T, self.V_U])

        self.u = tf.compat.v1.placeholder(tf.int32,[None,])
        self.i = tf.compat.v1.placeholder(tf.int32,[None,])

        self.rating = tf.compat.v1.placeholder(tf.float32, [None,])

        self.use_cnn = use_cnn
        if self.use_cnn:
            self.imgs_feats = tf.compat.v1.placeholder(tf.float32, [None, self.fc_feat_size])
            self.encode_img_W = tf.compat.v1.get_variable('encode_img_W', [self.fc_feat_size, self.image_emb_size], initializer=self.emb_initializer)
            self.encode_img_b = tf.compat.v1.get_variable('encode_img_b', [self.image_emb_size], initializer=self.const_initializer)

        self.item_bias_rnn = tf.Variable(tf.zeros([self.V_M]))
        self.user_bias_rnn = tf.Variable(tf.zeros([self.V_U]))

        self.learning_rate = learning_rate

        self.beta1 = beta1
        self.lamda = lamda  # regularization parameters
        self.initdelta = initdelta
        self.grad_clip = grad_clip
        self.update_rule = update_rule
        self.joint_loss = 0.0
        self.weight_initializer = tf.compat.v1.random_uniform_initializer(minval=-self.initdelta, maxval=self.initdelta,dtype=tf.float32)
        self.emb_initializer = tf.compat.v1.random_uniform_initializer(minval=-self.initdelta, maxval=self.initdelta,dtype=tf.float32)
        self.const_initializer = tf.compat.v1.constant_initializer(0.0)

    def _init_MF(self):
        with tf.compat.v1.variable_scope('MF'):
            if self.param_mf is None:
                self.mf_user_embeddings = tf.compat.v1.get_variable('mf_user_emb', [self.V_U, self.mf_emb_dim], initializer=self.emb_initializer)
                self.mf_item_embeddings = tf.compat.v1.get_variable('mf_item_emb', [self.V_M, self.mf_emb_dim], initializer=self.emb_initializer)
                self.mf_item_bias = tf.Variable(tf.zeros([self.V_M]))
                self.mf_user_bias = tf.Variable(tf.zeros([self.V_U]))
            else:
                self.mf_user_embeddings = tf.Variable(self.param[0])
                self.mf_item_embeddings = tf.Variable(self.param[1])
                self.mf_user_bias = tf.Variable(self.param[2])
                self.mf_item_bias = tf.Variable(self.param[3])

    def _decode_lstm(self, h_usr, h_itm, reuse=False):
        if False:
            with tf.compat.v1.variable_scope('D_rating', reuse=reuse):
                w_usr = tf.compat.v1.get_variable('w_usr', [self.rnn_size, self.rnn_size], initializer=self.weight_initializer)
                w_itm = tf.compat.v1.get_variable('w_itm', [self.rnn_size, self.rnn_size], initializer=self.weight_initializer)
#                bias = tf.get_variable('bias', [self.rnn_size], initializer=self.const_initializer)
                usr_vec = tf.matmul(h_usr, w_usr)
                itm_vec = tf.matmul(h_itm, w_itm)

                logits_RNN = tf.reduce_sum(input_tensor=tf.multiply(usr_vec, itm_vec), axis=1)
                self.paras_rnn.extend([w_usr,w_itm])
                return logits_RNN
        else:
            i_bias_rnn = tf.gather(self.item_bias_rnn, self.i)
            u_bias_rnn = tf.gather(self.user_bias_rnn, self.u)
            logits_RNN = tf.reduce_sum(input_tensor=tf.multiply(h_usr, h_itm), axis=1) + i_bias_rnn + u_bias_rnn
            print("Do not use a fully-connectted layer at the time of output decoding.")
            return logits_RNN

    def _get_initial_lstm(self, batch_size):
        with tf.compat.v1.variable_scope('D_initial_lstm'):
            c_itm = tf.zeros([batch_size, self.rnn_size], tf.float32)
            h_itm = tf.zeros([batch_size, self.rnn_size], tf.float32)
            c_usr = tf.zeros([batch_size, self.rnn_size], tf.float32)
            h_usr = tf.zeros([batch_size, self.rnn_size], tf.float32)
            # self.paras_rnn.extend([c_itm, h_itm, c_usr, h_usr])   # these variable should be trainable or not
            return c_itm, h_itm, c_usr, h_usr

    def _rnn_item_embedding(self, inputs, reuse=False):
        with tf.compat.v1.variable_scope('D_item_embedding', reuse=reuse):
            w = tf.compat.v1.get_variable('w', [self.V_U, self.input_emb_dim], initializer=self.emb_initializer)
            x_flat = tf.reshape(inputs, [-1, self.V_U]) #(N * T, U)
            x = tf.matmul(x_flat, w) #(N * T, H)
            x = tf.reshape(x, [-1, self.T, self.input_emb_dim]) #(N, T, H)
            self.paras_rnn.extend([w])
            return x

    def _rnn_user_embedding(self, inputs, reuse=False):
        with tf.compat.v1.variable_scope('D_user_embedding', reuse=reuse):
           w = tf.compat.v1.get_variable('w', [self.V_M, self.input_emb_dim], initializer=self.emb_initializer)
           x_flat = tf.reshape(inputs, [-1, self.V_M]) #(N * T, M)
           x = tf.matmul(x_flat, w) #(N * T, H)
           x = tf.reshape(x, [-1, self.T, self.input_emb_dim]) #(N, T, H)
           self.paras_rnn.extend([w])
           return x

    def all_logits(self,u):
        u_embedding = tf.nn.embedding_lookup(params=self.mf_user_embeddings, ids=u)
        return tf.matmul(u_embedding, self.mf_item_embeddings, transpose_a=False,transpose_b=True)+ self.mf_item_bias #+u_bias

    def get_mf_logists(self,u,i):
        mf_u_embedding = tf.nn.embedding_lookup(params=self.mf_user_embeddings, ids=u)
        mf_i_embedding = tf.nn.embedding_lookup(params=self.mf_item_embeddings, ids=i)
        mf_i_bias = tf.gather(self.mf_item_bias, i)
        mf_u_bias = tf.gather(self.mf_user_bias, u)
        logits_MF = tf.reduce_sum(input_tensor=tf.multiply(mf_i_embedding, mf_u_embedding), axis=1) + mf_i_bias + mf_u_bias
        return logits_MF

    def _attention_layer(self, features, features_proj, h, L,name, reuse=False):
        with tf.compat.v1.variable_scope(name+'_attention_layer', reuse=reuse):
            w = tf.compat.v1.get_variable('w', [self.mf_emb_dim, self.mf_emb_dim], initializer=self.weight_initializer)
            b = tf.compat.v1.get_variable('b', [self.mf_emb_dim], initializer=self.const_initializer)
            w_att = tf.compat.v1.get_variable('w_att', [self.mf_emb_dim, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.mf_emb_dim]), w_att), [-1, L])   # (N, L)
            alpha = tf.nn.softmax(out_att)

            context = tf.reduce_sum(input_tensor=features * tf.expand_dims(alpha, 2), axis=1, name='context')   #(N, D)
            return context, alpha

    def _project_features(self, features, L, name):
        with tf.compat.v1.variable_scope(name + '_project_features'):
            w = tf.compat.v1.get_variable('w', [self.mf_emb_dim, self.mf_emb_dim], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.mf_emb_dim])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, L, self.mf_emb_dim])
            return features_proj

    #initialiing al the rrquired variables before running discriminator and building the structure of model
    def build_pretrain(self):
        self._init_MF()
        batch_size = tf.cast(tf.shape(input=self.item_sequence)[0], tf.int32)

        c_itm, h_itm, c_usr, h_usr = self._get_initial_lstm(batch_size)
        itm_lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)
        usr_lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)

        input_itms = self._rnn_item_embedding(inputs=self.item_sequence)
        input_usrs = self._rnn_user_embedding(inputs=self.user_sequence)

        mf_u_embedding = tf.nn.embedding_lookup(params=self.mf_user_embeddings, ids=self.u)
        mf_i_embedding = tf.nn.embedding_lookup(params=self.mf_item_embeddings, ids=self.i)

        #rnn for item
        for t in range(self.T):
            with tf.compat.v1.variable_scope('G_itm_lstm', reuse=(t!=0)):
                if t == 0 and self.use_cnn and True:
                    current_emb = image_emb
                else:
                    n_t = t
                    if self.mixture == 'soft_V1':
                        current_emb = input_itms[:,n_t,:]
                    elif self.mixture == 'soft_V2':
                        current_emb = tf.concat( [input_itms[:,n_t,:], mf_i_embedding],axis=1)
                    elif self.mixture == 'soft_V3':
                        item_context,alpha = self._attention_layer(item_features,item_features_proj, h_itm, self.V_M, name='item', reuse=(t!=0))
                        current_emb = tf.concat([x_itm[:,n_t,:], item_context],axis=1)
                    elif self.mixture == 'hard':
                        current_emb = input_itms[:,n_t,:]
                    else:
                        assert False
                _, (c_itm, h_itm) = itm_lstm_cell(inputs=current_emb, state=[c_itm, h_itm])

        #rnn for user
        for t in range(self.T):
            with tf.compat.v1.variable_scope('G_usr-lstm', reuse=(t!=0)):
                if self.mixture == 'soft_V1':
                    current_emb = input_usrs[:,t,:]
                elif self.mixture == 'soft_V2':
                    current_emb = tf.concat( [input_usrs[:,t,:], mf_u_embedding],axis=1)
                elif self.mixture == 'soft_V3':
                    user_context,alpha = self._attention_layer(user_features, user_features_proj, h_usr, self.V_U, name='user', reuse=(t!=0))
                    current_emb = tf.concat([x_usr[:,t,:], user_context],axis=1)
                elif self.mixture == 'hard':
                    current_emb = input_usrs[:,t,:]
                else:
                    assert False
                _, (c_usr, h_usr) = usr_lstm_cell(inputs=current_emb, state=[c_usr, h_usr])
                # if self.mixture == 'soft_V3':
                #     logits_RNN = self._decode_lstm(h_usr, h_itm, reuse=False)
                #
                #     joint_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating[:,t], logits=logits_RNN)
                #     loss += tf.reduce_sum(input_tensor=joint_loss)

        self.logits_RNN = tf.nn.relu(self._decode_lstm(h_usr, h_itm, reuse=False)) #+self.i_bias_rnn
        self.logits_MF = tf.nn.relu(self.get_mf_logists(self.u,self.i))
        self.joint_logits = self.logits_MF + self.logits_RNN

        if self.mixture == 'soft_V1':
            self.loss_RNN = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.logits_RNN))
            self.loss_MF = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.logits_MF))
            #self.joint_loss = loss / tf.cast(batch_size, dtype=tf.float32)

        self.joint_loss += self.lamda * tf.reduce_sum(input_tensor=[ tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() ])

        if self.update_rule == 'adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.compat.v1.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer
        else:
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer
        global_step = tf.compat.v1.train.get_global_step()
        self.pretrain_updates = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1)\
                .minimize(self.joint_loss, var_list=tf.compat.v1.trainable_variables())

        self.all_logits = self.all_logits(self.u)

    def pretrain_step(self, sess, ratings, u, i,user_sequence=None, item_sequence=None, img_feats=None):

        if user_sequence is not None:
                outputs = sess.run([self.pretrain_updates, self.loss_MF ,self.loss_RNN,
                                    self.joint_loss,self.logits_MF,self.logits_RNN ],
                                    feed_dict = {self.user_sequence: user_sequence,
                                                 self.item_sequence: item_sequence,
                                                 self.rating: ratings,
                                                 self.u: u,
                                                 self.i: i,})
        #else:
            #print("phase12")
            #outputs = sess.run([self.pretrain_updates, self.joint_loss,self.pre_logits_MF], feed_dict = {self.rating: rating, self.u: u, self.i: i})

        return outputs

    def prediction(self, sess, user_sequence, item_sequence, u, i,sparse=True, use_sparse_tensor = None,img_feats=None):

        if sparse:
            #print("#%^^&%*^")
            user_sequence,item_sequence=[ii.toarray() for ii in user_sequence],[ii.toarray() for ii in item_sequence]

        if self.use_cnn:
            outputs = sess.run(self.joint_logits, feed_dict = {self.user_sequence: user_sequence,
                        self.item_sequence: item_sequence, self.u: u, self.i: i, self.imgs_feats:img_feats})
        else:
            #print("12345678")
            outputs = sess.run(self.joint_logits, feed_dict = {self.user_sequence: user_sequence,
                        self.item_sequence: item_sequence, self.u: u, self.i: i})

        return outputs

    def predictionItems(self, sess, u):
        outputs = sess.run(self.all_logits, feed_dict = {self.u: [u]})
        return outputs

    def getRewards(self,sess,gen, samples,sparse=False):
        u_seq,i_seq = [[ sample[i].toarray()  for sample in samples ]  for i in range(2)]
        u,i = [[ sample[i]  for sample in samples ]  for i in range(2,4)]

        labeled_rewards = np.zeros(len(samples))

        unlabeled_rewards = self.prediction(sess,u_seq,i_seq,u,i)

        rewards = labeled_rewards + unlabeled_rewards

        return 2 * (self.sigmoid(rewards) - 0.5)

    def saveMFModel(self, sess, filename):
        self.paras_mf = [self.mf_user_embeddings,self.mf_item_embeddings,self.mf_user_bias,self.mf_item_bias]
        param = sess.run(self.paras_mf)
        pickle.dump(param, open(filename, 'wb'))
