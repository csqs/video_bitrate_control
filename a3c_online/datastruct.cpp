#include <stdlib.h>  
#include <stdio.h>  

#include <iostream>
#include <vector>
#include <fstream> 
#include <string>

#include "datastruct.h"

std::string report_states_to_string(ReportInfor report_states_one)
{
   std::string report_states_string;

   report_states_string += "client_id:";
   report_states_string += std::to_string(report_states_one.client_id);
   report_states_string += spilt_note;

   report_states_string += "model_index:";
   report_states_string += std::to_string(report_states_one.model_index);
   report_states_string += spilt_note;

   report_states_string += "send_bitrate:";
   report_states_string += std::to_string(report_states_one.send_bitrate);
   report_states_string += spilt_note;

   report_states_string += "loss_rate:";
   report_states_string += std::to_string(report_states_one.loss_rate);
   report_states_string += spilt_note;

   report_states_string += "reward:";
   report_states_string += std::to_string(report_states_one.reward);
   report_states_string += spilt_note;

   return report_states_string;
}

std::string recv_qos_to_string(RecvQos recv_qos_one)
{
   std::string recv_qos_string;

   recv_qos_string += "next_send_bitrate:";
   recv_qos_string += std::to_string(recv_qos_one.next_send_bitrate);
   recv_qos_string += spilt_note;

   recv_qos_string += "model_index:";
   recv_qos_string += std::to_string(recv_qos_one.model_index);
   recv_qos_string += spilt_note;

   return recv_qos_string;
}

std::vector<float> split_string_to_data(std::string report_states_string)
{
   std::string::size_type pos1, pos2;
   pos2 = report_states_string.find(spilt_note);
   pos1 = 0;
   std::vector<float> data_in_string;
   while(std::string::npos != pos2)
   {

      std::string one_record_in_struct = report_states_string.substr(pos1, pos2-pos1);
      std::string::size_type pos3 = one_record_in_struct.find(":") + 1;

      std::string data_in_record = one_record_in_struct.substr(pos3);
      data_in_string.push_back(::atof(data_in_record.c_str()));

      pos1 = pos2 + spilt_note.size();
      pos2 = report_states_string.find(spilt_note, pos1);
   }
   return data_in_string;
}

ReportInfor string_to_report_states(std::string report_states_string){
   ReportInfor report_states_one;
   std::vector<float> data_in_string = split_string_to_data(report_states_string);

   report_states_one.client_id = data_in_string[0];
   report_states_one.model_index = data_in_string[1];
   report_states_one.send_bitrate = data_in_string[2];
   report_states_one.loss_rate = data_in_string[3];
   report_states_one.reward = data_in_string[4];

   return report_states_one;
}

RecvQos string_to_recv_qos(std::string recv_qos_string){
   RecvQos recv_qos_one;
   std::vector<float> data_in_string = split_string_to_data(recv_qos_string);

   recv_qos_one.next_send_bitrate = data_in_string[0];
   recv_qos_one.model_index = data_in_string[1];

   return recv_qos_one;
}



ExperienceQueue experience_queue_init(int client_id)
{
   ExperienceQueue experience_queue_one;
   experience_queue_one.client_id = float(client_id);
   experience_queue_one.model_index = 0.0;

   std::vector<Experience> client_experience_queue_one;
   Experience model_experience_one;
   model_experience_one.model_action = 0.0;
   model_experience_one.env_reward = 0.0;
   model_experience_one.model_index = 0.0;
   model_experience_one.entropy = 0.0;

   State state_history_cur_one;
   for(int k = 0; k < STATE_HISTORY_LENGTH; k++)
   {
      ReportInfor report_state_one;
      report_state_one.client_id = float(client_id);
      report_state_one.model_index = 0.0;
      report_state_one.send_bitrate = 0.0;
      report_state_one.loss_rate = 0.0;
      report_state_one.reward = 0.0;
      state_history_cur_one.report_info_queue.push_back(report_state_one);//S1=[s1 s2 s3...s8]
   }
   model_experience_one.state = state_history_cur_one;//S1
   client_experience_queue_one.push_back(model_experience_one);

   experience_queue_one.client_experience_queue_vec = client_experience_queue_one;
   return experience_queue_one;
}

void experience_queue_reset_to_zeros(ExperienceQueue& experience_queue_cur, int client_id)
{
    std::vector<Experience>().swap(experience_queue_cur.client_experience_queue_vec);//first clear

   ExperienceQueue experience_queue_one;
   experience_queue_one.client_id = float(client_id);
   experience_queue_one.model_index = 0.0;

   std::vector<Experience> client_experience_queue_one;
   Experience model_experience_one;
   model_experience_one.model_action = 0.0;
   model_experience_one.env_reward = 0.0;
   model_experience_one.entropy = 0.0;

   State state_history_cur_one;
   for(int k = 0; k < STATE_HISTORY_LENGTH; k++)
   {
      ReportInfor report_state_one;
      report_state_one.client_id = float(client_id);
      report_state_one.model_index = 0.0;
      report_state_one.send_bitrate = 0.0;
      report_state_one.loss_rate = 0.0;
      report_state_one.reward = 0.0;
      state_history_cur_one.report_info_queue.push_back(report_state_one);//S1=[s1 s2 s3...s8]
   }
   model_experience_one.state = state_history_cur_one;//S1
   client_experience_queue_one.push_back(model_experience_one);

   experience_queue_cur.client_experience_queue_vec = client_experience_queue_one;

}


std::deque<ReportInfor> state_history_queue_update(std::deque<ReportInfor> &report_info_queue, ReportInfor report_states_arrival)
{
   report_info_queue.pop_front();
   report_info_queue.push_back(report_states_arrival);
   return report_info_queue;
}

ExperienceQueue experience_queue_increase(ExperienceQueue& experience_queue_cur, ReportInfor report_states_arrival, float model_action, float env_reward){  
   Experience model_experience_new = experience_queue_cur.client_experience_queue_vec.back();//SM ->SM+1
   model_experience_new.model_action = model_action;
   model_experience_new.env_reward = env_reward;
   model_experience_new.model_index = report_states_arrival.model_index;

   model_experience_new.state.report_info_queue = state_history_queue_update(model_experience_new.state.report_info_queue, report_states_arrival);//SM ->SM+1

   experience_queue_cur.client_experience_queue_vec.push_back(model_experience_new);//[S1 S2 ...SM] -> [S1 S2 ...SM SM+1]
   experience_queue_cur.model_index = model_experience_new.model_index;

   return experience_queue_cur;
}

void experience_queue_increase_with_state(ExperienceQueue& experience_queue_cur,float reward, uint32_t model_index, float model_action, float entropy, State curstate)
{  
   if(experience_queue_cur.model_index < model_index)
   {
      experience_queue_reset_with_state(experience_queue_cur, reward, model_index, model_action, entropy, curstate);
   }
   else if(experience_queue_cur.model_index == model_index)
   {//
      Experience model_experience_new;
      model_experience_new.model_action = model_action;
      model_experience_new.env_reward = reward;
      model_experience_new.model_index = model_index;

      model_experience_new.state.report_info_queue = curstate.report_info_queue;//SM ->SM+1

      experience_queue_cur.client_experience_queue_vec.push_back(model_experience_new);//[S1 S2 ...SM] -> [S1 S2 ...SM SM+1]
      experience_queue_cur.model_index = model_index;
   }
   else
   {//err
      experience_queue_reset_with_state(experience_queue_cur, reward, model_index, model_action, entropy, curstate);
   }

}

void  experience_queue_reset_with_state(ExperienceQueue& experience_queue_cur, float reward, uint32_t model_index, float model_action, float entropy, State curstate)
{  
   Experience model_experience_new;
   model_experience_new.model_action = model_action;
   model_experience_new.env_reward = reward;
   model_experience_new.model_index = model_index;
   model_experience_new.entropy = entropy;
   //Sm -> Sm+1 model_experience_new
   model_experience_new.state.report_info_queue = curstate.report_info_queue;//SM ->SM+1
   
   // experience_queue_cur.client_experience_queue.clear();//clear the vector and pushback Sm+1
   std::vector<Experience>().swap(experience_queue_cur.client_experience_queue_vec);

   experience_queue_cur.client_experience_queue_vec.push_back(model_experience_new);//Sm
   experience_queue_cur.model_index = model_index;
}


State generate_state(ExperienceQueue& experience_queue_cur, ReportInfor report_states_arrival)
{
   Experience model_experience_prev = experience_queue_cur.client_experience_queue_vec.back();//SM ->SM+1
   State cur_state;
   cur_state.report_info_queue = state_history_queue_update(model_experience_prev.state.report_info_queue, report_states_arrival);//SM ->SM+1
   return cur_state;
}

std::vector<ExperienceQueue> model_input_init(){
   std::vector<ExperienceQueue> all_clients_experience_queues;

   for(int i = 0; i < NUM_AGENTS; i++){
      ExperienceQueue experience_queue_one = experience_queue_init(i);
      experience_queue_one.client_id = float(i);

      all_clients_experience_queues.push_back(experience_queue_one);
   }
   return all_clients_experience_queues;
}

void epoch_log_record_to_txt(epoch_log_record epoch_log_record_one)
{
   std::ofstream ofresult("train_log.txt", std::ios::app); 

   std::string log_line = "model_index:" + std::to_string(int(epoch_log_record_one.model_index)) + "\n" 
                          "avg_reward:" + std::to_string(epoch_log_record_one.avg_reward) + "\n"
                          "avg_td_error:" + std::to_string(epoch_log_record_one.avg_td_error) + "\n"
                          "avg_entropy:" + std::to_string(epoch_log_record_one.avg_entropy) + "\n";

   ofresult << log_line << std::endl;
}
