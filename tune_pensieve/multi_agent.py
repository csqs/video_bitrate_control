#coding:utf-8
import os
import logging
import numpy as np
import multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import env
import a3c
import load_trace
import test_predict

DEFAULT_QUALITY = 1  # default video quality without agent
S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 16  # take how many frames in the past
A_DIM = 12

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]  # Kbps


BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 49.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
#NN_MODEL = './results_pre/pretrain_linear_reward.ckpt'
NN_MODEL = None

def central_agent(net_params_queues, exp_queues, epoch_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    assert len(epoch_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=10000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0
        reward_mean_cur =0
        reward_mean_br =0
        reward_mean_rebuf = 0
        reward_mean_smooth = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                epoch_queues[i].put(epoch)
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic, epoches_num=epoch)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))


            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy,
                summary_vars[3]: reward_mean_cur
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                reward_mean_cur = test_predict.TestRun(sess, actor, critic, epoch)


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, epoch_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # 1.从center同步最新的模型参数 initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        epoch_num = epoch_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM) #初始化 动作空间A个actions
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []


        time_stamp = 0
        while True:  # experience video streaming forever

            # 和环境Env交互 the action is from the last decision
            # this is to make the framework similar to the real
            # delay, sleep_time, buffer_size, rebuf, \
            # video_chunk_size, next_video_chunk_sizes, \
            # end_of_video, video_chunk_remain = \
            #     net_env.get_video_chunk(bit_rate)

            assert bit_rate >= 0
            assert bit_rate < A_DIM
            bitrate_send_last, lossrate_recv_last, bitrate_real_recovery,\
            bitrate_send_last_probe, lossrate_recv_last_probe, bitrate_real_recovery_probe,\
             end_of_video \
            = net_env.action_dispatch_and_report_svr(VIDEO_BIT_RATE[bit_rate])

            time_stamp += 2
            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
	    #print '1', net_env.netbw
	    #print '2', bitrate_send_last_probe * (1 - lossrate_recv_last_probe)
            x_funtion_top = (bitrate_send_last_probe * (1 - lossrate_recv_last_probe) - VIDEO_BIT_RATE[bit_rate])  / M_IN_K
            reward = -x_funtion_top * x_funtion_top # 0.1 0.2 ... 1.1 1.2

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            #state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            #state[0, -1] = bitrate_send_last / 1000.0  # last quality
            #state[1, -1] = lossrate_recv_last  # 丢包率0.1 0.2 0.3 0.4
            #state[2, -1] = bitrate_real_recovery / 1000.0  # kilo byte / ms

            state = np.roll(state, -1, axis=1)
            state[0, -1] = bitrate_send_last_probe / 1000.0  # last quality
            state[1, -1] = lossrate_recv_last_probe  # 丢包率0.1 0.2 0.3 0.4
            state[2, -1] = bitrate_real_recovery_probe / 1000.0  # kilo byte / ms

            state[3, :A_DIM] = np.array(VIDEO_BIT_RATE[:]) / 1000.0  # kilo byte / ms
            state[4, -1] = bitrate_send_last / 1000.0  # kilo byte / ms
            # print state[3, :A_DIM]

            # ================== Predict BandWidth =========================

            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)

            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(bitrate_send_last) + '\t' +
                           str(lossrate_recv_last) + '\t' +
                           str(bitrate_real_recovery) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                epoch_num = epoch_queue.get()

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main():
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    epoches_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))
        epoches_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, epoches_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i], epoches_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
