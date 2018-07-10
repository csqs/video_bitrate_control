#coding:utf-8
import numpy as np


RANDOM_SEED = 42
LINK_RTT = 80  # millisec

NetBW =[150, 300, 600, 1200, 2400] #5
# NetLossRate =[0, 0.15, 0.25, 0.45] #4
NetLossRate =[0] #4
TimePeriodArray= [30, 45, 60, 75] #4
TimePeriodArray= [30] #4
ProbeBwFactor = 1.5

class Environment:
    def __init__(self,  random_seed=RANDOM_SEED):
        np.random.seed(random_seed)

        # pick a random trace file
        self.netbw_idx = 0
        self.netbw = NetBW[self.netbw_idx]
        self.net_lossrate_index = 0
        self.net_lossrate = NetLossRate[self.net_lossrate_index]
        self.videochattime_index =0
        self.videochat_time = TimePeriodArray[self.videochattime_index]
        self.video_chattime_counter =0

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
                bitrate_exp_recovery = 1.0 *  bitrate_send_last*(1-net_lossrate) / (1.0 + fec_rate) #实际收到的码率 乘以恢复率
            else:
                net_lossrate_real = 1.0 * (bitrate_send_last - net_bw *(1 - net_lossrate)) / (1.0 * bitrate_send_last)
                fec_rate = 3 * net_lossrate_real
                bitrate_exp_recovery = 1.0 *  net_bw * (1 - net_lossrate) / (1.0 + fec_rate)
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


    def action_dispatch_and_report_svr(self, BitRateQos):
        bitrate_send_last = BitRateQos
        bitrate_send_probe = BitRateQos * ProbeBwFactor
	#bitrate_send_probe = self.netbw
        lossrate_recv_last = 0
        end_of_video = False
        end_of_validation = False
        net_bw = self.netbw
        net_lossrate = self.net_lossrate

        bitrate_real_recovery, lossrate_recv_last = self.env_reaction_to_bitrate_adjust(bitrate_send_last, net_bw, net_lossrate)
        bitrate_real_recovery_probe, lossrate_recv_last_probe = self.env_reaction_to_bitrate_adjust(bitrate_send_probe, net_bw, net_lossrate)
        self.video_chattime_counter +=1
        # print 'videochat_time:',self.videochat_time, ' cur:',self.video_chattime_counter
        if self.video_chattime_counter >= self.videochat_time:
            end_of_video = True
            self.videochattime_index +=1
            if self.videochattime_index== len(TimePeriodArray):
                self.videochattime_index=0
                self.net_lossrate_index +=1
                if self.net_lossrate_index == len(NetLossRate):
                    self.net_lossrate_index =0
                    self.netbw_idx +=1
                    if self.netbw_idx == len(NetBW):
                        self.netbw_idx = 0
                        end_of_validation = True

            self.netbw = NetBW[self.netbw_idx]
            self.net_lossrate = NetLossRate[self.net_lossrate_index]
            self.videochat_time = TimePeriodArray[self.videochattime_index]
            self.video_chattime_counter = 0

        # if end_of_video == True:
        #     print 'NetBw LossRate VideoTime:', self.netbw_idx, ' ',self.net_lossrate_index, ' ',self.videochattime_index

        return bitrate_send_last, lossrate_recv_last, bitrate_real_recovery, \
               bitrate_send_probe, lossrate_recv_last_probe, bitrate_real_recovery_probe, \
               end_of_video, end_of_validation



