#coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import fixed_env as env
import a3c
import load_trace
#import matplotlib.pyplot as plt

RANDOM_SEED = 42
RAND_RANGE = 1000
M_IN_K = 1000.0
DEFAULT_QUALITY = 1  # default video quality without agent
S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 16  # take how many frames in the past
A_DIM = 12
VIDEO_BIT_RATE = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]  # Kbps

def TestRun(sess, actor, critic, epoch):

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM


    net_env = env.Environment()


    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    reward_sum_all = []
    reward_video_all = []
    reward_sum_per_video = []

    reward_mean_cur = 0



    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        assert bit_rate >= 0
        assert bit_rate < A_DIM

        bitrate_send_last, lossrate_recv_last, bitrate_real_recovery, \
        bitrate_send_last_probe, lossrate_recv_last_probe, bitrate_real_recovery_probe, \
        end_of_video, end_of_validation \
            = net_env.action_dispatch_and_report_svr(VIDEO_BIT_RATE[bit_rate])

        time_stamp += 2  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = bitrate_real_recovery / M_IN_K  # 0.1 0.2 ... 1.1 1.2

        r_batch.append(reward)

        last_bit_rate = bit_rate
        reward_sum_per_video.append(reward)


        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record

        state = np.roll(state, -1, axis=1)
        # this should be S_INFO number of terms
        state[0, -1] = bitrate_send_last / 1000.0  # last quality
        state[1, -1] = lossrate_recv_last  # 丢包率0.1 0.2 0.3 0.4
        state[2, -1] = bitrate_real_recovery / 1000.0  # kilo byte / ms

        state = np.roll(state, -1, axis=1)
        state[0, -1] = bitrate_send_last_probe / 1000.0  # last quality
        state[1, -1] = lossrate_recv_last_probe  # 丢包率0.1 0.2 0.3 0.4
        state[2, -1] = bitrate_real_recovery_probe / 1000.0  # kilo byte / ms

        state[3, :A_DIM] = np.array(VIDEO_BIT_RATE[:]) / 1000.0  # kilo byte / ms
        state[4, -1] = bitrate_send_last / 1000.0  # kilo byte / ms


        action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        # log_file.write('action_prob: '+ str(action_prob)+'\n')
        action_cumsum = np.cumsum(action_prob)
        # log_file.write('action_cumsum: ' + str(action_cumsum)+'\n')
        random_value = np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)
        decision_arrary = (action_cumsum > random_value)
        bit_rate = decision_arrary.argmax()
        # log_file.write('decision: ' + str(bit_rate) + ' random value: ' + str(random_value) + ' decision_arrary: ' + str(decision_arrary)+'\n')
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append(state)

        entropy_record.append(a3c.compute_entropy(action_prob[0]))

        if end_of_video:

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            reward_sum_all.append(reward_sum_per_video[1:])
            video_reward_sum = np.sum(reward_sum_per_video[1:])
            reward_video_all.append(video_reward_sum)
            meanvalue = np.mean(reward_sum_per_video)
            stdvalue = np.mean(reward_sum_per_video)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []
            reward_sum_per_video = []

            # print "video count", video_count, 'video_reward_sum:%.3f', video_reward_sum, ' meanvalue:', meanvalue, ' stdvalue:',stdvalue
            # print ("video count: %d video_reward_sum:%.3f meanvalue:%.3f stdvalue:%.3f"%(video_count, video_reward_sum, meanvalue, stdvalue))



            if end_of_validation:
                mean_all_video_reward = np.mean(reward_video_all)
                sum_all_video_reward = np.sum(reward_video_all)
                std_all_video_reward = np.std(reward_video_all)
                reward_mean_cur = mean_all_video_reward
                # print ("video total count: %d reward_sum:%.3f reward_mean:%.3f reward_std:%.3f" % (
                #     video_count, sum_all_video_reward, mean_all_video_reward, std_all_video_reward))

                print 'epoch:', epoch, ' reward_mean: ', reward_mean_cur
                break
    return reward_mean_cur






