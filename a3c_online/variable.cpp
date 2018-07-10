
#include "variable.h"

int g_client_num = 0; 
uint32_t NUM_AGENTS = 1;

int MAX_NUM_REPORT = 5;

std::string spilt_note = "|";

int STATE_HISTORY_LENGTH = 8;
int EXPERIENCE_QUEUE_LENGETH = 3;
int NUM_EXPERIENCE_INPUT = 2;

uint32_t NUM_OBSERVATIONS = 3;
uint32_t NUM_LENGTH = 8;
uint32_t NUM_ACTIONS = 12;
float GAMMA = 0.99f;
uint32_t MIN_TRAIN_EXPERIENCE_LEN=5;

std::vector<float> VIDEO_BIT_RATE {100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0};

int EW_UPDATE_EPOCH = 10000;
int STOP_EPOCH = 120000;
float INIT_ENTROPY_WEIGHT = 5.0;
