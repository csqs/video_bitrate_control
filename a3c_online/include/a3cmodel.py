#coding:utf-8
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf
import tflearn

GAMMA = 0.99

#ENTROPY_WEIGHT = 1
ENTROPY_EPS = 1e-6

S_INFO = 3  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 12
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
SUMMARY_DIR = './models'

class NetworkVP(object):
    GRAPH_FILEPATH = 'models/graph_a3c'

    def __init__(self):

        self.x_states = tf.placeholder(tf.float32, shape=(None, S_INFO*S_LEN), name='x_states')
        #self.x_states = tf.placeholder(tf.float32, shape=(None, S_INFO), name='x_states')
        self.y_acts = tf.placeholder(tf.float32, [None, A_DIM], name='y_acts')
        self.y_td_target = tf.placeholder(tf.float32, shape=(None, 1), name='y_td_target')
        self.y_td_error = tf.placeholder(tf.float32, [None, 1], name='y_td_error')

        self.entropy_weight = tf.placeholder(tf.float32, [1, 1], name='entropy_weight')

        self.x_states_reshape = tf.reshape(self.x_states, [-1, 3, 8])

        actor_split_0 = tflearn.conv_1d(self.x_states_reshape[:, 0:1, :], 128, 4, activation='relu')
        actor_split_1= tflearn.conv_1d(self.x_states_reshape[:, 1:2, :], 128, 4, activation='relu')
        actor_split_2= tflearn.conv_1d(self.x_states_reshape[:, 2:3, :], 128, 4, activation='relu')

        # actor network
        '''
        actor_split_0 = tflearn.conv_1d(self.x_states[:,0:1: 0:8], 128, 4, activation='relu')
        actor_split_1= tflearn.conv_1d(self.x_states[:, 0:1, S_LEN:(2*S_LEN)], 128, 4, activation='relu')
        actor_split_2= tflearn.conv_1d(self.x_states[:, 0:1, (2*S_LEN):(3*S_LEN)], 128, 4, activation='relu')
        '''

        '''
        self.x_states = tf.placeholder(tf.float32, shape=(None, S_INFO, S_LEN), name='x_states')
        #self.x_states = tf.placeholder(tf.float32, shape=(None, S_INFO), name='x_states')
        self.y_acts = tf.placeholder(tf.float32, [None, A_DIM], name='y_acts')
        self.y_td_target = tf.placeholder(tf.float32, shape=(None, 1), name='y_td_target')
        self.y_td_error = tf.placeholder(tf.float32, [None, 1], name='y_td_error')

        # actor network
        actor_split_0 = tflearn.conv_1d(self.x_states[:, 0:1, :], 128, 4, activation='relu')
        actor_split_1= tflearn.conv_1d(self.x_states[:, 1:2, :], 128, 4, activation='relu')
        actor_split_2= tflearn.conv_1d(self.x_states[:, 2:3, :], 128, 4, activation='relu')
        '''

        actor_split_0_flat = tflearn.flatten(actor_split_0)
        actor_split_1_flat = tflearn.flatten(actor_split_1)
        actor_split_2_flat = tflearn.flatten(actor_split_2)

        actor_merge_net = tflearn.merge([actor_split_0_flat, actor_split_1_flat, actor_split_2_flat], 'concat')

        actor_dense_net_0 = tflearn.fully_connected(actor_merge_net, 128, activation='relu')
        self.out_policies = tflearn.fully_connected(actor_dense_net_0, A_DIM, activation='softmax', name='out_policies')

        # critic network
        '''
        critic_split_0 = tflearn.conv_1d(self.x_states[:, 0:1, :], 128, 4, activation='relu')
        critic_split_1= tflearn.conv_1d(self.x_states[:, 1:2, :], 128, 4, activation='relu')
        critic_split_2= tflearn.conv_1d(self.x_states[:, 2:3, :], 128, 4, activation='relu')
        '''

        critic_split_0 = tflearn.conv_1d(self.x_states_reshape[:, 0:1, :], 128, 4, activation='relu')
        critic_split_1= tflearn.conv_1d(self.x_states_reshape[:, 1:2, :], 128, 4, activation='relu')
        critic_split_2= tflearn.conv_1d(self.x_states_reshape[:, 2:3, :], 128, 4, activation='relu')

        '''
        critic_split_0 = tflearn.conv_1d(self.x_states[:, 0:S_LEN], 128, 4, activation='relu')
        critic_split_1= tflearn.conv_1d(self.x_states[:, S_LEN:(2*S_LEN)], 128, 4, activation='relu')
        critic_split_2= tflearn.conv_1d(self.x_states[:, (2*S_LEN):(3*S_LEN)], 128, 4, activation='relu')
        '''

        critic_split_0_flat = tflearn.flatten(critic_split_0)
        critic_split_1_flat = tflearn.flatten(critic_split_1)
        critic_split_2_flat = tflearn.flatten(critic_split_2)

        critic_merge_net = tflearn.merge([critic_split_0_flat, critic_split_1_flat, critic_split_2_flat], 'concat')

        critic_dense_net_0 = tflearn.fully_connected(critic_merge_net, 128, activation='relu')
        self.out_values = tflearn.fully_connected(critic_dense_net_0, 1, activation='linear', name='out_values')

        # actor train
        self.actor_obj = tf.reduce_sum(tf.multiply(
                       tf.log(tf.reduce_sum(tf.multiply(self.out_policies, self.y_acts),
                                            reduction_indices=1, keep_dims=True)),
                       -self.y_td_error)) \
                   + self.entropy_weight * tf.reduce_sum(tf.multiply(self.out_policies,
                                                           tf.log(self.out_policies + ENTROPY_EPS)))

        self.actor_optimize = tf.train.RMSPropOptimizer(ACTOR_LR_RATE)
        self.actor_optimize.minimize(self.actor_obj, name='actor_minimize')

        # critic train
        self.critic_obj = tflearn.mean_square(self.y_td_target, self.out_values)
        self.critic_optimize = tf.train.RMSPropOptimizer(CRITIC_LR_RATE)
        self.critic_optimize.minimize(self.critic_obj, name='critic_minimize')

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def save(self, path=GRAPH_FILEPATH):
        tf.train.Saver().save(self.session, path)


if __name__ == '__main__':

    model = NetworkVP()
    model.save()





















