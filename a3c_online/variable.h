#ifndef VARIABLE_H
#define VARIABLE_H

#define IQNAME "/states_in"
#define OQNAME "/action_out"

#include <stdlib.h>  
#include <stdio.h> 
#include <iostream>
#include <vector>

extern int g_client_num; 
extern uint32_t NUM_AGENTS;

extern int MAX_NUM_REPORT;

extern std::string spilt_note;

extern int STATE_HISTORY_LENGTH;
extern int EXPERIENCE_QUEUE_LENGETH;
extern int NUM_EXPERIENCE_INPUT;

extern uint32_t NUM_OBSERVATIONS;
extern uint32_t NUM_LENGTH;
extern uint32_t NUM_ACTIONS;
extern float GAMMA;

extern std::vector<float> VIDEO_BIT_RATE;

extern int EW_UPDATE_EPOCH;
extern int STOP_EPOCH;
extern float INIT_ENTROPY_WEIGHT;
extern uint32_t MIN_TRAIN_EXPERIENCE_LEN;

#endif

