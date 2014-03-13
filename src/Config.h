#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "basic_func.h"

using namespace std;

class Config{
public:
    unsigned int num_user;
    unsigned int num_item;
    unsigned int num_rating;
    float rt_lambda;
    double rt_lrate;
    float sgd_lambda;
    float sgd_lrate;
    float bias_reg;
    unsigned int num_factor;
    //number of user extra feature
    unsigned int num_user_feature;
    //number of item extra feature
    unsigned int num_item_feature;
    unsigned int m_max_epochs;
    //The leaf number on a tree to prevent overfitting
    unsigned int m_max_tree_leafs;
    //Randomly select part of feature for every single tree trainning
    //Stochastic gradient boosting
    unsigned int m_feature_subspace_size;
    bool m_use_opt_splitpoint;
    //shrinkage/learning rate
    //Randomly select ratio of feature for every single tree trainning
    //Stochastic gradient boosting
    float m_data_sample_ratio; 

private:
    string config_file_name;
    void valueAssignment(const string& name, const string& val){
        if(name == "num_user") num_user = atoi(val.c_str());
        else if(name == "num_item") num_item = atoi(val.c_str());
        else if(name == "num_rating") num_rating = atoi(val.c_str());
        else if(name == "num_factor") num_factor = atoi(val.c_str());
        else if(name == "num_user_feature") num_user_feature = atoi(val.c_str());
        else if(name == "num_item_feature") num_item_feature = atoi(val.c_str());
        else if(name == "m_max_epochs") m_max_epochs = atoi(val.c_str());
        else if(name == "m_max_tree_leafs") m_max_tree_leafs = atoi(val.c_str());
        else if(name == "m_feature_subspace_size") m_feature_subspace_size = atoi(val.c_str());
        else if(name == "sgd_lrate") sgd_lrate = strtof(val.c_str(), NULL);
        else if(name == "bias_reg") bias_reg = strtof(val.c_str(), NULL);
        else if(name == "rt_lrate") rt_lrate = strtof(val.c_str(), NULL);
        else if(name == "m_data_sample_ratio") m_data_sample_ratio = strtof(val.c_str(), NULL);
        else if(name == "sgd_lambda") sgd_lambda= strtof(val.c_str(), NULL);
        else if(name == "rt_lambda") rt_lambda= strtof(val.c_str(), NULL);
    }
public:
    Config(const string& file_name){
        config_file_name = file_name;
        num_user = 0;
        num_item = 0;
        num_rating = 0;
        num_factor = 32;
        num_user_feature = 0;
        num_item_feature = 0;
        m_max_epochs = 300;
        m_max_tree_leafs = 20;
        m_feature_subspace_size = 20;
        m_use_opt_splitpoint = true;
        bias_reg = 0.0;
        sgd_lrate = 0.1;
        rt_lrate = 0.1;
        m_data_sample_ratio = 0.5; 
        sgd_lambda = 0.0;
        rt_lambda = 0.0;
    }
    void PrintConf(){
        cout << "config_file_name:" << config_file_name << endl;
        cout << "num_user:" << num_user << endl;
        cout << "num_item:" << num_item << endl;
        cout << "num_rating:" << num_rating << endl;
        cout << "num_factor:" << num_factor << endl;
        cout << "num_user_feature:" << num_user_feature << endl;
        cout << "num_item_feature:" << num_item_feature << endl;
        cout << "m_max_epochs:" << m_max_epochs << endl;
        cout << "m_max_tree_leafs:" << m_max_tree_leafs << endl;
        cout << "m_feature_subspace_size:" << m_feature_subspace_size << endl;
        cout << "sgd_lrate:" << sgd_lrate << endl;
        cout << "rt_lrate:" << rt_lrate << endl;
        cout << "rt_lambda:" << rt_lambda << endl;
        cout << "sgd_lambda:" << sgd_lambda << endl;
        cout << "bias_reg:" << bias_reg << endl;
        cout << "m_data_sample_ratio:" << m_data_sample_ratio<< endl;
    }
    void LoadConfig(){
    /*
     * Load configuration from config file
     * Format:
     *  configName = configValue
     * if a line start with '#', it was a comment line
     */
        ifstream fs;
        fs.open(config_file_name.c_str(), std::ios::in);
        if (fs.fail()){
            cerr << "config file was not exist!!" << endl;
            exit(-1);
        }
        string line, name, val;
        unsigned int line_num = 0;
        while(getline(fs, line)){
            if(!line.empty()){
                line_num += 1;
                if(line[0] == '#') continue;
                if(Split(line, name, val)){
                    valueAssignment(name, val);
                }
                else{
                    cerr << "Configure format wrong in line " << line_num << endl;
                }
            }
        }
    }
};
#endif
