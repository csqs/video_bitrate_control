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

#include <deque>

#include "datastruct.h"

void* pthread_client_report_states_then_recv_qos (void* arg)  
{  
   int *pClient_id  = (int *)arg;
   int client_id = *pClient_id;
   printf("Client_%d thread_client_report_states\n", client_id);
   printf ("This is client:%d: report thread!\n", client_id);
   if(client_id>100)
   {
      printf("Error CLient ID\n");
      exit(1);
      return NULL;
   }
   mqd_t mqd_input, mqd_output;
   int ret;
   char input_buf[BUFSIZ], output_buf[BUFSIZ];

   mqd_input = mq_open(IQNAME, O_RDWR | O_CREAT, 0600, NULL);
   std::string outputqueue_name = OQNAME;
   outputqueue_name += std::to_string(client_id);
   mqd_output = mq_open(outputqueue_name.c_str(), O_RDWR | O_CREAT, 0600, NULL);

   if (mqd_input == -1 || mqd_output == -1)
   {
      perror("mqd_input || mqd_output : mq_open()\n");
      exit(1);
      return NULL;
   }

   ReportInfor report_infor_one;
   report_infor_one.client_id = float(client_id);
   report_infor_one.send_bitrate = 0.0;
   report_infor_one.model_index = 0.0;
   report_infor_one.loss_rate = 0.1;
   report_infor_one.reward = 1.0;

   for(int i=0; i < MAX_NUM_REPORT; i++)
   { 
      //std::string input_buf_string = std::to_string(client_id);
      
      report_infor_one.loss_rate = 0.1;
      report_infor_one.reward = i+1;
      report_infor_one.terminalflag = false;
      if(i==MAX_NUM_REPORT-1)
         report_infor_one.terminalflag = true;

      std::string input_buf_string = report_states_to_string(report_infor_one);

      printf("client_%d : send message [%s] to center svr...\n", client_id, input_buf_string.c_str());
      // ret = mq_send(mqd_input, input_buf_string.c_str(), BUFSIZ, 1);
      ret = mq_send(mqd_input, (char*)&report_infor_one, sizeof(report_infor_one), 1);

      if (ret == -1)
      {
         perror("error: agent send message to center svr\n");
         continue;
      }
      else
      {
         for(int i=0; i < MAX_NUM_REPORT; i++)
         {  

            RecvQos qos_recv;
            ret = mq_receive(mqd_output, (char *)&qos_recv, BUFSIZ, NULL);
            if(ret == -1)
            {
               perror("error: client receive message from center\n");
               continue;
            }
            else
            {
               report_infor_one.model_index = qos_recv.model_index;
               report_infor_one.send_bitrate = qos_recv.next_send_bitrate;
               printf("client_%u : next_send_bitrate[%f]  from center...\n", qos_recv.model_index, qos_recv.next_send_bitrate);
               break;
            }       
         }   
         sleep (2);
      }
   } 

   mq_close(mqd_input);
   mq_close(mqd_output);

   // mq_unlink(IQNAME);
   mq_unlink(outputqueue_name.c_str());
   printf("CLientId:%d thread exit!\n", client_id);

   pthread_exit (NULL); 
   // return NULL;
}