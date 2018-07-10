

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "a3cmodel.h"
#include "../variable.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"


std::vector<float> one_hot_encode(int action)
{
    std::vector<float> encoded_action(NUM_ACTIONS, 0.0f);
    encoded_action[action] = 1.0f;
    return encoded_action;
}


Model::Model():
    session(tf::NewSession( {}))
{
    TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), META_GRAPH_FILEPATH, &meta_graph_def));
    TF_CHECK_OK(session->Create(meta_graph_def.graph_def()));

    // restore variables values
    auto graph_filepath_tensor = tf::Tensor(tf::DT_STRING, {});
    graph_filepath_tensor.scalar<tf::string>()() = GRAPH_FILEPATH;

    std::vector<std::pair<tf::string, tf::Tensor>> inputs = 
    {
        {meta_graph_def.saver_def().filename_tensor_name(), graph_filepath_tensor}
    };
    TF_CHECK_OK(session->Run(inputs, {}, {meta_graph_def.saver_def().restore_op_name()}, nullptr));
}

Model::~Model(){}

void Model::fit(TrainQueue input_queue_one, float model_entropy_weight)
{
    // fill the states tensor
    //std::cout << "states_tensor" << std::endl;
    epoch_log_record epoch_log_record_one;
    epoch_log_record_one.model_index = input_queue_one.model_index;
    epoch_log_record_one.avg_reward = 0.0;
    epoch_log_record_one.avg_td_error = 0.0;
    epoch_log_record_one.avg_entropy = 0.0;
    
    for(auto input_index = 0; input_index < input_queue_one.all_clients_train_queue.size(); input_index ++)
    {
        ExperienceQueue experience_queue_one = input_queue_one.all_clients_train_queue[input_index];

        auto experience_queue_length = experience_queue_one.client_experience_queue_vec.size();
        if(experience_queue_length <= MIN_TRAIN_EXPERIENCE_LEN)
            continue;

        auto states_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(experience_queue_length), NUM_OBSERVATIONS * NUM_LENGTH});
        auto states_eigenmatrix = states_tensor.matrix<float>();
        bool bIsTerminal = experience_queue_one.bIsTerminal;

        for(auto i = 1; i < experience_queue_length -1; i++)//we drop the first data and the last data 
        { //0] 1 2 3 ...29]30
            auto model_experience_one = experience_queue_one.client_experience_queue_vec[i];
            auto state = model_experience_one.state;
            int32_t state_index = 0;

            for(auto state_index_in_window = 0; state_index_in_window < state.report_info_queue.size(); state_index_in_window++)
            {
                auto state_one = state.report_info_queue[state_index_in_window];
                states_eigenmatrix(i, state_index) = state_one.send_bitrate;
                state_index ++;
            }

            for(auto state_index_in_window = 0; state_index_in_window < state.report_info_queue.size(); state_index_in_window++)
            {
                auto state_one = state.report_info_queue[state_index_in_window];
                states_eigenmatrix(i, state_index) = state_one.loss_rate;
                state_index ++;
            }

            for(auto state_index_in_window = 0; state_index_in_window < state.report_info_queue.size(); state_index_in_window++)
            {
                auto state_one = state.report_info_queue[state_index_in_window];
                states_eigenmatrix(i, state_index) = state_one.reward;
                state_index ++;
            }
        }

        // fill the rewards tensor
        //std::cout << "td_target_tensor" << std::endl;
        uint32_t train_len = experience_queue_length-2;
        auto td_target_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(train_len), 1});
        auto td_target_eigenmatrix = td_target_tensor.matrix<float>();

        td_target_eigenmatrix((train_len - 1), 0) = 0;
        uint32_t experienct_num_acc = 0;
        if(!bIsTerminal)
        {
            auto model_experience_one = experience_queue_one.client_experience_queue_vec[experience_queue_length-1];
            auto env_reward = model_experience_one.env_reward;
            td_target_eigenmatrix((train_len - 1), 0) = env_reward;
        }
        float avg_reward = 0.0;
        float avg_entropy = 0.0;

        for (auto i = (experience_queue_length - 2); i >= 2; i--) 
        {
            auto model_experience_one = experience_queue_one.client_experience_queue_vec[i];
            auto env_reward = model_experience_one.env_reward;
            td_target_eigenmatrix(i-2, 0) = env_reward + GAMMA * td_target_eigenmatrix(i-2+1, 0);

            avg_reward += env_reward;
            avg_entropy += model_experience_one.entropy;

            experienct_num_acc ++;
        }

        epoch_log_record_one.avg_reward += avg_reward / experienct_num_acc;
        epoch_log_record_one.avg_entropy += avg_entropy / experienct_num_acc;

        // train the model
        std::vector<std::pair<tf::string, tf::Tensor>> critic_inputs = 
        {
            {"x_states", states_tensor}, {"y_td_target", td_target_tensor}
        };//0] 1 2 ....29]30 /

        //std::cout << "outputs" << std::endl;
        std::vector<tf::Tensor> outputs;

        TF_CHECK_OK(session->Run(critic_inputs, {"out_values/BiasAdd:0"}, {}, &outputs));//V(s)
        TF_CHECK_OK(session->Run(critic_inputs, {}, {"critic_minimize:0"}, nullptr));

        //std::cout << "out_value_eigentensor" << std::endl;
        const auto out_value_eigentensor = outputs[0].flat<float>();
        std::vector<float> out_value(out_value_eigentensor.size());
        for (auto i = 0; i < out_value_eigentensor.size(); ++i) 
        {
            out_value[i] = out_value_eigentensor(i);
            //std::cout << i << ":" << out_value[i] << std::endl;
        }
        //std::cout << "td_error_tensor" << std::endl;
        auto td_error_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(train_len), 1});
        auto td_error_eigenmatrix = td_error_tensor.matrix<float>();
        // td_error_eigenmatrix((experience_queue_length - 2), 0) = 0;
        float avg_td_error = 0.0;
        for (auto i = 0; i < train_len; ++i) 
        {
            td_error_eigenmatrix(i, 0) = td_target_eigenmatrix(i, 0) - out_value[i];
            avg_td_error += td_error_eigenmatrix(i, 0);
        }
        epoch_log_record_one.avg_td_error += avg_td_error / train_len;

        // fill the actions tensor
        //std::cout << "actions_tensor" << std::endl;
        auto actions_tensor = tf::Tensor(tf::DT_FLOAT, {static_cast<long long>(train_len), NUM_ACTIONS});
        auto actions_eigenmatrix = actions_tensor.matrix<float>();
        for (auto i = 0; i < train_len; ++i) //we drop the first data and the last data  
        {
            auto model_experience_one = experience_queue_one.client_experience_queue_vec[i+1];//1 2 3... 29 (len-2)
            const auto action = one_hot_encode(int(model_experience_one.model_action));
            for (auto j = 0; j < action.size(); ++j) 
            {
                actions_eigenmatrix(i, j) = action[j];
            }
        }

        auto entropy_weight = tf::Tensor(tf::DT_FLOAT, {1, 1});
        auto entropy_weight_eigenmatrix = entropy_weight.matrix<float>();
        entropy_weight_eigenmatrix(0, 0) = model_entropy_weight;
        std::vector<std::pair<tf::string, tf::Tensor>> actor_inputs = 
        {
            {"x_states", states_tensor}, {"y_acts", actions_tensor}, {"y_td_error", td_error_tensor}, {"entropy_weight", entropy_weight}
        };
        TF_CHECK_OK(session->Run(actor_inputs, {}, {"actor_minimize:0"}, nullptr));
    }

    epoch_log_record_one.avg_reward = epoch_log_record_one.avg_reward / input_queue_one.all_clients_train_queue.size();
    epoch_log_record_one.avg_td_error = epoch_log_record_one.avg_td_error / input_queue_one.all_clients_train_queue.size();
    epoch_log_record_one.avg_entropy = epoch_log_record_one.avg_entropy / input_queue_one.all_clients_train_queue.size();
    epoch_log_record_to_txt(epoch_log_record_one);
}


std::vector<float> Model::predict_policy(State state_history_one)
{
    auto states_tensor = tf::Tensor(tf::DT_FLOAT, {1, NUM_OBSERVATIONS * NUM_LENGTH});
    std::vector<float> state;
    for(auto i = 0; i < state_history_one.report_info_queue.size(); i++)
    {
        ReportInfor report_states_one = state_history_one.report_info_queue[i];
        state.push_back(report_states_one.send_bitrate);
    }
    for(auto i = 0; i < state_history_one.report_info_queue.size(); i++)
    {
        ReportInfor report_states_one = state_history_one.report_info_queue[i];
        state.push_back(report_states_one.loss_rate);
    }
    for(auto i = 0; i < state_history_one.report_info_queue.size(); i++)
    {
        ReportInfor report_states_one = state_history_one.report_info_queue[i];
        state.push_back(report_states_one.reward);
    }
    std::copy_n(state.cbegin(), state.size(), states_tensor.flat<float>().data());

    std::vector<std::pair<tf::string, tf::Tensor>> inputs = 
    {
        {"x_states", {states_tensor}}
    };
    
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(session->Run(inputs, {"out_policies/Softmax:0"}, {}, &outputs));

    const auto out_policy_eigentensor = outputs[0].flat<float>();
    std::vector<float> out_policy(out_policy_eigentensor.size());
    for (auto i = 0; i < out_policy.size(); ++i) 
    {
        out_policy[i] = out_policy_eigentensor(i);
        std::cout << i << ":" << out_policy[i] << std::endl;
    }
    const auto probabilities_sum = std::accumulate(out_policy.cbegin(), out_policy.cend(), 0.0f);
    assert(probabilities_sum > 0.99f and probabilities_sum < 1.01f);

    return out_policy;
}

void Model::save()
{
    auto graph_filepath_tensor = tf::Tensor(tf::DT_STRING, {});
    graph_filepath_tensor.scalar<tf::string>()() = GRAPH_FILEPATH;

    std::vector<std::pair<tf::string, tf::Tensor>> inputs = 
    {
        {"save/Const:0", graph_filepath_tensor}
    };
    TF_CHECK_OK(session->Run(inputs, {}, {"save/control_dependency:0"}, nullptr));
}

/*
summary_ops, summary_vars = a3c.build_summaries()
writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })
writer.add_summary(summary_str, epoch)
writer.flush()




def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars



 */

