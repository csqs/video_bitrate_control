#ifndef DATASTRUCT_H
#define DATASTRUCT_H

#include <stdlib.h>  
#include <stdio.h> 

#include <iostream>
#include <vector>
#include <deque>

#include "variable.h"

//client report environment states to server
struct ReportInfor{
   uint32_t client_id;
   uint32_t model_index;
   float send_bitrate;
   float loss_rate;
   float reward;
   bool terminalflag;
   ReportInfor()
   {
      client_id = 0;
      model_index = 0;
      send_bitrate = 0;
      loss_rate = 0;
      reward = 0;
      terminalflag = false;
   }
};

//server notify client the next action
struct RecvQos{
   float next_send_bitrate;
   uint32_t model_index;
};

/*
data prepared before one epoch a3c training
---
store states as state maxtrix => State
---
states matrix to critor => model_action  
<states matrix, model_action, env_reward> => Experience
---
experinences needs when training an a3c model one time => ExperienceQueue
---
inputs need when training a model one epoch => TrainQueue
---
 */

struct State{
   std::deque<ReportInfor> report_info_queue;
};//3x8 循环移位

struct Experience{ //St =[si si+1 si+2 ... si+n-1]]n=8
   uint32_t model_index;
   State state;
   float model_action;
   float env_reward;//暂时没有
   float entropy;
};

struct ExperienceQueue{
   uint32_t client_id;
   uint32_t model_index;
   bool bIsTerminal;
   std::vector<Experience> client_experience_queue_vec;//30个经验 S1 S2...S30
};//经验队列 

struct TrainQueue{
   uint32_t model_index;
   std::vector<ExperienceQueue> all_clients_train_queue;//这个达到一定值送去训练
};

struct epoch_log_record{
   uint32_t model_index;
   float avg_reward;
   float avg_td_error;
   float avg_entropy;
};

//match the message queue requirements
std::string report_states_to_string(ReportInfor report_states_one);
std::string recv_qos_to_string(RecvQos recv_qos_one);

/*
parse the message from the message queue
split message as : "client_id:0.0|send_rate:0.0|loss_rate:0.0|recv_rate:0.0|"
 */
std::vector<float> split_string_to_data(std::string report_states_string);
ReportInfor string_to_report_states(std::string report_states_string);
RecvQos string_to_recv_qos(std::string recv_qos_string);

//init one client experience queue
ExperienceQueue experience_queue_init(int client_id);
//init all the client experience queues in the server
std::vector<ExperienceQueue> model_input_init();
/*
when server receives a state report from the client message queue
the client experience queue increases
new experience <= advantage <= State update
 */
std::deque<ReportInfor> state_history_queue_update(std::deque<ReportInfor> &report_info_queue, ReportInfor report_states_arrival);
ExperienceQueue experience_queue_increase(ExperienceQueue& experience_queue_cur, ReportInfor report_states_arrival, float model_action, float entropy, float env_reward);

State generate_state(ExperienceQueue& experience_queue_cur, ReportInfor report_states_arrival);
void  experience_queue_reset_with_state(ExperienceQueue& experience_queue_cur, float reward, uint32_t model_index, float model_action, float entropy, State curstate);
void experience_queue_increase_with_state(ExperienceQueue& experience_queue_cur,float reward, uint32_t model_index, float model_action, float entropy, State curstate);
void experience_queue_reset_to_zeros(ExperienceQueue& experience_queue_cur, int client_id);
void epoch_log_record_to_txt(epoch_log_record epoch_log_record_one);

#endif
