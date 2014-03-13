#ifndef __TASKER_H__
#define __TASKER_H__

#include "gbdt.h"
#include "matrix.h"

class SVDFeatureTasker{
public:
    SVDFeatureTasker();
    SVDFeatureTasker(map<string, string>& argv);
public:
    void Train(const int& n_round);
    void Update();
    void Predict();
    void TrainInit();
    void SaveModel(const int& round_num);
    void LoadModel(const int& round_num);
    void PredictInit(const int& n_round);
    void CalculateRMSE();
    void CrossValidation();

private:
    Config* config;
    FactorMatrix* U;
    FactorMatrix* V;
    BiasFactor* bias_factor;
    YMatrix* y_matrix;

private:
    string config_file;
    string train_file;
    string predict_file;
    string user_feature_file;
    string cross_validation_file;
    string item_feature_file;
    string DEBUG_FIEL;
};

#endif
