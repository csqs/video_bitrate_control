#include <stdlib.h>  
#include <stdio.h>  
#include <pthread.h>  
#include <unistd.h>
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

#include "variable.h"

extern void* pthread_svr_handle_report (void* arg);
extern void* pthread_client_report_states_then_recv_qos (void* arg);

int main (int argc, char** argv)  
{  
   int ret = 0;  
   //Message Queue 
   //We first Create one Input queue(states_in Queue) and N OutputQueue(action_out Queue)
   std::vector<mqd_t> mqd_output_queue;
   mqd_t mqd_input;
   char input_buf[BUFSIZ], output_buf[BUFSIZ];
  
   mq_unlink(IQNAME);
   mqd_input = mq_open(IQNAME, O_RDWR | O_CREAT, 0600, NULL);
   if (mqd_input == -1)
   {
      perror("mqd_input : mq_open()\n");
      exit(-1);
   }
   uint32_t prio;

   for (int i = 0; i < NUM_AGENTS; i++)
   {
      mqd_t mqd_output;
      std::string outputqueue_name = OQNAME;
      outputqueue_name += std::to_string(i);
      mq_unlink(outputqueue_name.c_str());
      mqd_output = mq_open(outputqueue_name.c_str(), O_RDWR | O_CREAT, 0644, NULL);
      if (mqd_output == -1)
      {
         perror("mqd_output main: mq_open()\n");
         exit(-1);
      }
      else
      {
         //while((mq_receive(mqd_output, output_buf, BUFSIZ, &prio)) != -1);
         mqd_output_queue.push_back(mqd_output);
      }
   }

   for(int i=0; i < NUM_AGENTS; i++)
   {
      pid_t pid = fork();
      if(pid == 0)
      {
         pthread_t pt_1 = 0;    
         ret = pthread_create(&pt_1, NULL, pthread_client_report_states_then_recv_qos,(void*)&g_client_num);  
         if (ret != 0)
         {  

            perror ("pthread_client_report_states_then_recv_qos fail \n");  
            exit(-1);
         }
         else
         {
            printf("pthread:%d create  succeed!\n", g_client_num);
         }
         pthread_join (pt_1, NULL);
         pthread_exit(NULL);   
      }
      else
      {}
      g_client_num++;
   }

   pthread_t pt_1 = 0;
   ret = pthread_create(&pt_1, NULL, pthread_svr_handle_report, NULL);  
   if (ret != 0)
   {  
      perror ("pthread_svr_handle_report fail \n");
      exit(-1);  
   }
   else
   {
      printf("pthread_svr_handle_report create succeed!\n");

   }
   pthread_join (pt_1, NULL);
   // pthread_exit(NULL);

   int queue_size = mqd_output_queue.size();
   printf("MainProcess MessageQueue size:%d \n", mqd_output_queue.size());
   for(int i=0;i< mqd_output_queue.size();i ++)
   {
      mq_close(mqd_output_queue[i]);

   }
   std::vector<mqd_t>().swap(mqd_output_queue);
   printf("MainProcess Release MessageQueue size:%d \n", mqd_output_queue.size());
   mq_unlink(IQNAME);
   for (int i = 0; i < NUM_AGENTS; i++)
   {
      std::string outputqueue_name = OQNAME;
      outputqueue_name += std::to_string(i);
      mq_unlink(outputqueue_name.c_str());
      printf("MainProcess MessageQueue Close [%d] name:%s\n", i, outputqueue_name.c_str());
   }

   // exit(0);
   pthread_exit(NULL); 
   exit(0); 
   return 0;  
}
