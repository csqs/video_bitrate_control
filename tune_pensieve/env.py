#coding:utf-8
import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 12
TOTAL_VIDEO_CHUNCK = 49
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'

NetBW =[100, 200, 500, 1000, 1500, 2000, 5000]
# NetLossRate =[0, 0.05, 0.10, 0.20, 0.30, 0.50]
NetLossRate =[0]
TimePeriodMin = 30
TimePeriodMax = 90
ProbeBwFactor = 1.5


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        # pick a random trace file
        self.netbw_idx = np.random.randint(len(NetBW))
        self.netbw = NetBW[self.netbw_idx]
        self.net_lossrate_index = np.random.randint(len(NetLossRate))
        self.net_lossrate = NetLossRate[self.net_lossrate_index]
        self.videochat_time = np.random.randint(TimePeriodMin, TimePeriodMax + 1)
        self.video_chattime_counter =0


    def action_dispatch_and_report_svr(self, BitRateQos):
        bitrate_send_last = BitRateQos 
        bitrate_send_probe = BitRateQos * ProbeBwFactor
	#bitrate_send_probe = self.netbw
        end_of_video = False
        net_bw = self.netbw
        net_lossrate = self.net_lossrate

        bitrate_real_recovery, lossrate_recv_last = self.env_reaction_to_bitrate_adjust(bitrate_send_last, net_bw, net_lossrate)
        bitrate_real_recovery_probe, lossrate_recv_last_probe = self.env_reaction_to_bitrate_adjust(bitrate_send_probe, net_bw, net_lossrate)
        self.video_chattime_counter +=1
        # print 'videochat_time:',self.videochat_time, ' cur:',self.video_chattime_counter
        if self.video_chattime_counter >= self.videochat_time:
            end_of_video = True

            # pick a random trace file
            self.netbw_idx = np.random.randint(len(NetBW))
            self.netbw = NetBW[self.netbw_idx]
            self.net_lossrate_index = np.random.randint(len(NetLossRate))
            self.net_lossrate = NetLossRate[self.net_lossrate_index]
            self.videochat_time = np.random.randint(TimePeriodMin, TimePeriodMax + 1)
            self.video_chattime_counter = 0

        return bitrate_send_last, lossrate_recv_last, bitrate_real_recovery,\
               bitrate_send_probe, lossrate_recv_last_probe, bitrate_real_recovery_probe, \
               end_of_video


    def env_reaction_to_bitrate_adjust(self, bitrate_send_last, net_bw, net_lossrate):
        assert bitrate_send_last > 0
        assert net_bw > 0
        assert net_lossrate >= 0
        assert net_lossrate < 0.6
        lossrate_recv_last = 0
        net_lossrate_real =0
        if net_lossrate == 0:
            if bitrate_send_last <= net_bw:
                bitrate_exp_recovery = bitrate_send_last
                net_lossrate_real =0
            else:
                net_lossrate_real = 1.0 * (bitrate_send_last - net_bw) / (1.0 * bitrate_send_last)
                fec_rate = 3 * net_lossrate_real
                bitrate_exp_recovery = 1.0 * net_bw / (1.0 + fec_rate) ##
            # print 'NetLoss=0 S and Bw: S=', bitrate_send_last, ' net_bw:', net_bw, ' net_lossrate=', net_lossrate, ' bitrate_exp=', bitrate_exp_recovery
        else:
            if bitrate_send_last <= net_bw:
            	net_lossrate_real = net_lossrate
                fec_rate = 3 * net_lossrate_real
                bitrate_exp_recovery = 1.0 * bitrate_send_last*(1-net_lossrate) / (1.0 + fec_rate) #实际收到的码率 乘以恢复率
            else:
                net_lossrate_real = 1.0 * (bitrate_send_last - net_bw *(1 - net_lossrate)) / (1.0 * bitrate_send_last)
                fec_rate = 3 * net_lossrate_real
                bitrate_exp_recovery = 1.0 * net_bw * (1 - net_lossrate) / (1.0 + fec_rate)
        lossrate_recv_last = net_lossrate_real
            # print 'NetLoss>0', 'net_lossrate=', net_lossrate, ' real_lossrate=', net_lossrate_real, ' S and Bw: S=', bitrate_send_last, ' net_bw:', net_bw, ' net_lossrate=', net_lossrate, ' bitrate_exp=', bitrate_exp_recovery
        # if bitrate_exp_recovery>net_bw*(1 - net_lossrate) and lossrate_recv_last>0.5:#
        if bitrate_exp_recovery>net_bw*(1 - net_lossrate):#
        	bitrate_exp_recovery =0
        bitrate_real_recovery = bitrate_exp_recovery
        if bitrate_real_recovery > net_bw:
            bitrate_real_recovery = net_bw
        # print ' bitrate_exp_recovery=', bitrate_exp_recovery, ' real_recovery=', bitrate_real_recovery, ' lossrate_recv_last=', lossrate_recv_last
        return bitrate_real_recovery, lossrate_recv_last
        
        
        



