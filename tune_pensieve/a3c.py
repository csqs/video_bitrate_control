#coding:utf-8
import numpy as np
import tensorflow as tf
import tflearn


GAMMA = 0.99
A_DIM = 12
ENTROPY_WEIGHT = 1
ENTROPY_EPS = 1e-6

class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.out = self.create_actor_network() #构建网络结构

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters  #操作网络节点的参数
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])  #需要Critic网络提供TD error
        self.entropy_factor = tf.placeholder(tf.float32, shape=())
        # Compute the objective (log action_vector and entropy) #优化目标 最小化
        self.obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.act_grad_weights)) \
                   + self.entropy_factor * tf.reduce_sum(tf.multiply(self.out,
                                                           tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here 计算梯度操作Op
        self.actor_gradients = tf.gradients(self.obj, self.network_params) #对目标函数求梯度

        # Optimization Op 应用梯度操作Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params)) #优化操作 将求解的梯度应用到网络参数上

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :A_DIM], 128, 4, activation='relu')
            split_4 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)


            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            return inputs, out

    def train(self, inputs, acts, act_grad_weights):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights, belta_entropy):

        # print 'belta_entropyin: ', belta_entropy,' entropy_factor:', self.sess.run(self.entropy_factor, feed_dict={self.entropy_factor:belta_entropy})
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights,
            self.entropy_factor : belta_entropy
        })



    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params) #计算梯度

        # Optimization Op 应用梯度去更新模型参数
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])


            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], 128, 4, activation='relu')
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], 128, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :A_DIM], 128, 4, activation='relu')
            split_4 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)


            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic, epoches_num):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert s_batch.shape[0] == a_batch.shape[0]
    assert s_batch.shape[0] == r_batch.shape[0]
    ba_size = s_batch.shape[0] #47

    v_batch = critic.predict(s_batch) #s状态是累积了当前agent用户连续的一个traces状态链 S1 S2 S3...ST

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(xrange(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch #TD error 可以认为是对每个动作的打分 advanatage

    #epoches_num = epoches_num*10000
    stage_step = 100
    cur_belta_entropy = 0.1
    multi_factor = 0
    if epoches_num<5*stage_step:
        belta_entropy = 5.0
        sub_value = 1
        multi_factor = epoches_num/stage_step #0 1 2 3 4
        cur_belta_entropy = belta_entropy - sub_value*multi_factor # 5 -0 -1 -2 -3 -4 =5 4 3 2 1
    elif epoches_num<8*stage_step:
    	belta_entropy = 1.0
        sub_value = 0.2
        multi_factor = epoches_num/stage_step - 4  #5 6 7 -4 = 1 2 3
        cur_belta_entropy = belta_entropy - sub_value*multi_factor # 1.0 -0.2 -0.4 -0.6 = 0.8 0.6 0.4 
    elif epoches_num<11*stage_step:
    	belta_entropy = 0.4
        sub_value = 0.1
        multi_factor = epoches_num/stage_step - 7  # 8 9 10 -7= 1 2 3
        cur_belta_entropy = belta_entropy - sub_value*multi_factor # 0.4 -0.1 -0.2 -0.3 = 0.3 0.2 0.1
    else:
    	cur_belta_entropy = 0.1


    if epoches_num % 100==0:
        print 'epoches_num: ', epoches_num, "multi_factor: ", multi_factor, ' cur_belta_entropy: ', cur_belta_entropy

    # cur_belta_entropy = 0.1
    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch, cur_belta_entropy)
    critic_gradients = critic.get_gradients(s_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in xrange(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    reward_mean = tf.Variable(0.)
    tf.summary.scalar("reward_mean", reward_mean)
    # reward_mean_br = tf.Variable(0.)
    # tf.summary.scalar("reward_mean_br", reward_mean_br)
    # reward_mean_rebuf = tf.Variable(0.)
    # tf.summary.scalar("reward_mean_rebuf", reward_mean_rebuf)
    # reward_mean_smooth = tf.Variable(0.)
    # tf.summary.scalar("reward_mean_smooth", reward_mean_smooth)

    # summary_vars = [td_loss, eps_total_reward, avg_entropy, reward_mean, reward_mean_br, reward_mean_rebuf, reward_mean_smooth ]
    summary_vars = [td_loss, eps_total_reward, avg_entropy, reward_mean]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
