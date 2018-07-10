#include <stdlib.h>  
#include <stdio.h>  
#include <pthread.h>  
#include <unistd.h>
#include <math.h>

#include <errno.h>  
#include <fcntl.h>
#include <sys/stat.h>        /* For mode constants */
#include <mqueue.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <deque>
#include <string>

#include "tools/easylogging++.h"
INITIALIZE_EASYLOGGINGPP

#include "datastruct.h"
#include "include/a3cmodel.h"

std::random_device random_device;
std::default_random_engine random_engine(random_device());

// logging after server receiver a message from the client
void report_states_to_log(ReportInfor client_report_infor)
{
   std::string log_line =  "server receives message from client " + std::to_string(int(client_report_infor.client_id)) + "\n" +
                          "the message details:\n" +
                          "client_id: " + std::to_string(int(client_report_infor.client_id)) + "\n" +
                          "model_index:" +  std::to_string(int(client_report_infor.model_index)) + "\n" +
                          "send_bitrate:" +  std::to_string(client_report_infor.send_bitrate) + "\n" +
                          "loss_rate:" +  std::to_string(client_report_infor.loss_rate) + "\n" +
                          "reward:" +  std::to_string(client_report_infor.reward) + "\n";
   //LOG(INFO) << log_line;
   std::cout << log_line << std::endl;
}

void recv_qos_to_log(RecvQos qos_dispatch_to_client, int client_id)
{
   std::string log_line =  "server send message to client: " + std::to_string(client_id) + "\n" +
                          "the message details:\n" +
                          "client_id: " + std::to_string(client_id) + "\n" +
                          "model_index:" +  std::to_string(int(qos_dispatch_to_client.model_index)) + "\n" +
                          "next_send_bitrate:" +  std::to_string(qos_dispatch_to_client.next_send_bitrate) + "\n";
   //LOG(INFO) << log_line;
   std::cout << log_line << std::endl;
}

//logging after new experience entered the client experience queue 
void client_queue_to_log(std::vector<Experience> &client_experience_queue_vec, int client_id)
{
   std::string log_line = "server experience queue from client " + std::to_string(client_id) + "\n" +
                         "the experience queue details:\n";

   for(int i = 0; i < client_experience_queue_vec.size(); i++)
   {
      log_line += "experience_index: " + std::to_string(i) + "\n";
      Experience model_experience_one = client_experience_queue_vec[i];

      log_line += "State: \n";
      State state_history_one = model_experience_one.state;

      log_line += "send_bitrate:";
      for(int j = 0; j < state_history_one.report_info_queue.size(); j++)
      {
         ReportInfor client_report_infor = state_history_one.report_info_queue[j];
         log_line += " " + std::to_string(client_report_infor.send_bitrate) + " ";
      }
      log_line += "\n";

      log_line += "loss_rate:";
      for(int j = 0; j < state_history_one.report_info_queue.size(); j++)
      {
         ReportInfor client_report_infor = state_history_one.report_info_queue[j];
         log_line += " " + std::to_string(client_report_infor.loss_rate) + " ";
      }
      log_line += "\n";

      log_line += "reward:";
      for(int j = 0; j < state_history_one.report_info_queue.size(); j++)
      {
         ReportInfor client_report_infor = state_history_one.report_info_queue[j];
         log_line += " " + std::to_string(client_report_infor.reward) + " ";
      }
      log_line += "\n";

      log_line += "model_action: " + std::to_string(model_experience_one.model_action) + "\n";
      log_line += "env_reward: " + std::to_string(model_experience_one.env_reward) + "\n";
   }
   //LOG(INFO) << log_line;
   std::cout << log_line << std::endl;
}

//logging after one model input entered the svr_input_queue
void input_queue_to_log(TrainQueue svr_input_queue)
{
   std::string log_line = "server collected experiences from client experience queue \n";
   log_line += "server inupt_queue length: " + std::to_string(svr_input_queue.all_clients_train_queue.size()) + "\n"; 
   //LOG(INFO) << log_line;
   std::cout << log_line << std::endl;
}

float cal_entropy(std::vector<float> out_policy){
   float H = 0.0;
   for(int i = 0; i < out_policy.size(); i++){
      float one_action_prob = out_policy[i];
      if(one_action_prob > 0 & one_action_prob < 1){
         H -= one_action_prob * log(one_action_prob); 
      }
   }
   return H;
}

void* pthread_svr_handle_report (void* arg)  
{  
   printf ("This is svr: report handle thread!\n");
   std::vector<mqd_t> mqd_output_queue;
   mqd_t mqd_input;
   int ret;
   uint32_t prio;
   char input_buf[BUFSIZ];

   /*
   set the message queue
      1 input queue, collect ReportInfor from all the clients
      NUM_AGENTS output queues, send RecvQos to each client
   */
   mqd_input = mq_open(IQNAME, O_RDWR);
   if (mqd_input == -1)
   {
      perror("mqd_input : mq_open()\n");
      exit(1);
   }

   for (int i = 0; i < NUM_AGENTS; i++)
   {
     mqd_t mqd_output;
     std::string outputqueue_name = OQNAME;
     outputqueue_name += std::to_string(i);
     mqd_output = mq_open(outputqueue_name.c_str(), O_RDWR);
     if (mqd_output == -1)
     {
        perror("mqd_output : mq_open()\n");
        exit(1);
     }
     else
     {
        mqd_output_queue.push_back(mqd_output);
     }
   }

   /*
   begin the message receive & send between the server and clients
    */
   //all the client experience queues in server, NUM_AGENTS * EXPERIENCE_QUEUE_LENGETH
   std::vector<ExperienceQueue> all_clients_experience_queues = model_input_init();
   //store NUM_EXPERIENCE_INPUT inputs for one epoch training 
   TrainQueue svr_input_queue;
   //model_index is a3c model training epoch number
   svr_input_queue.model_index = 0.0;
   int last_model_index = 0;
   float model_entropy_weight = INIT_ENTROPY_WEIGHT;

   /*
   load a3c model 
    */
   auto model = std::make_shared<Model>();

   for(int i=0; i < (MAX_NUM_REPORT * NUM_AGENTS); i++)
   {  
      printf("svr receive message from client...\n");
      int ret = 0;
      ReportInfor client_report_infor;
      // ret = mq_receive(mqd_input, input_buf, BUFSIZ, &prio);
      ret = mq_receive(mqd_input, (char *)&client_report_infor, BUFSIZ, NULL);
      if (ret == -1)
      {
         perror("error: center svr receive message from client\n");
         continue;
      }
      else
      {
         /*
         get a message from one client message queue
          */
         /*
         get the reward from the a3c based on report states
         default: set next_send_bitrate as the send_bitrate 
          */
         uint32_t client_id = client_report_infor.client_id;
         float env_reward = client_report_infor.reward;//Todo1 
         uint32_t model_index = client_report_infor.model_index;
         bool bIsTerminal = client_report_infor.terminalflag;
         printf("ReportInfor clientid:%u modelid:%u reward:%.3f terminal:%d \n", client_id, model_index, env_reward, bIsTerminal);

         //1. generate Sm+1 for Predict
        State state_cur = generate_state(all_clients_experience_queues[client_id], client_report_infor);
        std::vector<float> out_policy = model->predict_policy(state_cur);
        int action = std::discrete_distribution<int>(out_policy.cbegin(), out_policy.cend())(random_engine);
        float next_send_bitrate = VIDEO_BIT_RATE[action];
        float model_action = float(action);
        float entropy = cal_entropy(out_policy);
        
         //2. decide wheter this State_experience should be put into the Experience Queue
         //3. decide whether this Experience Queue should be put into the Train Queue
         //4. decide whether we can Update the model using the Train Queue

         if(client_report_infor.model_index == svr_input_queue.model_index)
         {

            experience_queue_increase_with_state(all_clients_experience_queues[client_id], env_reward, model_index, model_action, entropy, state_cur);

            client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);
            if(all_clients_experience_queues[client_id].client_experience_queue_vec.size() >= EXPERIENCE_QUEUE_LENGETH || bIsTerminal)
            {
              /*
               store the full client experience queue as one model input in svr_input_queue
               keep the svr_input_queue consistency on model_index
               */
                all_clients_experience_queues[client_id].bIsTerminal = bIsTerminal;
                if(all_clients_experience_queues[client_id].model_index == svr_input_queue.model_index)//Todo: end signal we shold reset to zeros
                {
                  svr_input_queue.all_clients_train_queue.push_back(all_clients_experience_queues[client_id]);//

                  input_queue_to_log(svr_input_queue);
                                    /*
                  begin one epoch model training when svr_input_queue is full
                  here the consistency problem comes from 
                  1. outdate message in message queue
                  2. outdate experience in client experience queue
                  */
                  if(svr_input_queue.all_clients_train_queue.size() >= NUM_EXPERIENCE_INPUT)
                  {
                                         /*
                     train the model
                     */
                     printf("model update :%d \n", svr_input_queue.model_index);
                     model->fit(svr_input_queue, model_entropy_weight);
                    
                     svr_input_queue.model_index += 1;
                     int moedl_index_now = int(svr_input_queue.model_index);
                     if(moedl_index_now > STOP_EPOCH)
                     {
                        model->save();

                        pthread_exit (0);    
                        return NULL;
                     }
                     else if((moedl_index_now - last_model_index) >= EW_UPDATE_EPOCH)
                     {
                        last_model_index = moedl_index_now;
                        if(moedl_index_now <= 50000)
                        {
                           model_entropy_weight -= 1.0;
                        }
                        else if(moedl_index_now <= 80000)
                        {
                           model_entropy_weight -= 0.2;
                        }
                        else if(moedl_index_now > 80000 & model_entropy_weight >= 0.1)
                        {
                           model_entropy_weight -= 0.1;
                        }

                     }
                     svr_input_queue.all_clients_train_queue.clear();//TrainQueue
                  }//end train queue
                }

                if(bIsTerminal)
                {
                  experience_queue_reset_to_zeros(all_clients_experience_queues[client_id], client_id);
                }
                else
                {
                  experience_queue_reset_with_state(all_clients_experience_queues[client_id], env_reward, svr_input_queue.model_index, model_action, entropy, state_cur);
                }
                client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);

            }//else do nothing
         }
         else if(uint32_t(client_report_infor.model_index) < svr_input_queue.model_index)
         {
            if(bIsTerminal)
            {
              experience_queue_reset_to_zeros(all_clients_experience_queues[client_id], client_id);
              client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);
            }
            else if(all_clients_experience_queues[client_id].model_index < svr_input_queue.model_index)
            {//clear and reset the experience queue
              experience_queue_reset_with_state(all_clients_experience_queues[client_id], env_reward, svr_input_queue.model_index, model_action, entropy, state_cur);
              client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);

            }
            else if(all_clients_experience_queues[client_id].model_index > svr_input_queue.model_index)
            {//error
              experience_queue_reset_with_state(all_clients_experience_queues[client_id], env_reward, svr_input_queue.model_index, model_action, entropy, state_cur);
              client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);
            }

         }
         else
         {//error
            if(bIsTerminal)
            {
              experience_queue_reset_to_zeros(all_clients_experience_queues[client_id], client_id);
              client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);
            }
            else
            {
              experience_queue_reset_with_state(all_clients_experience_queues[client_id], env_reward, svr_input_queue.model_index, model_action, entropy, state_cur);
              client_queue_to_log(all_clients_experience_queues[client_id].client_experience_queue_vec, client_id);
            }

         }



         /*
         update the next action from the a3c based on report states
         */
         /*
         store experience in the matching client experience queue
         keep the client experience queue consistency on model_index
          */
                
         /*
         send a message to one client message queue
          */
                                           
         RecvQos qos_dispatch_to_client;
         qos_dispatch_to_client.next_send_bitrate = next_send_bitrate;
         qos_dispatch_to_client.model_index = svr_input_queue.model_index;

         for(int i=0; i < MAX_NUM_REPORT; i++)
         {
            ret = mq_send(mqd_output_queue[client_id], (char*)&qos_dispatch_to_client, sizeof(qos_dispatch_to_client), 1);
            if (ret == -1)
            {
               perror("error: center send message to agent\n");
               continue;
            }
            else
            {
               recv_qos_to_log(qos_dispatch_to_client, client_id);
               break;
            }
         }  
      }
      sleep (1); //1 Sec
   }
   mq_close(mqd_input);

   printf("Svr MessageQueue size:%d \n", mqd_output_queue.size());
   for(int i=0;i< mqd_output_queue.size();i ++)
   {
      mq_close(mqd_output_queue[i]);

   }
   std::vector<mqd_t>().swap(mqd_output_queue);
   printf("Svr Release MessageQueue size:%d \n", mqd_output_queue.size());
   mq_unlink(IQNAME);
   for (int i = 0; i < NUM_AGENTS; i++)
   {
      std::string outputqueue_name = OQNAME;
      outputqueue_name += std::to_string(i);
      mq_unlink(outputqueue_name.c_str());
      printf("Svr MessageQueue Close [%d] name:%s\n", i, outputqueue_name.c_str());
   }

   printf("Svr thread exit!\n");
   pthread_exit (NULL);    
   // return NULL;
}
