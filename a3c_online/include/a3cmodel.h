#ifndef A3CMODEL_H
#define A3CMODEL_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../datastruct.h"

#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

class Model
{
	private:
        const std::string GRAPH_FILEPATH = "./include/models/graph_a3c";
        const std::string META_GRAPH_FILEPATH = "./include/models/graph_a3c.meta";

        tf::MetaGraphDef meta_graph_def;
        std::unique_ptr<tf::Session> session;

    public:
        Model();
		~Model();

        void fit(TrainQueue input_queue_one, float model_entropy_weight);
        std::vector<float> predict_policy(State state_history_one);
        //std::pair<std::vector<float>, float> predict_policy_and_value(const observation_t&);
        void save();
};


#endif // MODEL_H
