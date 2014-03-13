#include "tasker.h"
#include <sstream>
#include <math.h>

//Default constructor
SVDFeatureTasker::SVDFeatureTasker(){
    config = NULL;
    U = NULL;
    V = NULL;
    bias_factor = NULL;
    y_matrix = NULL;
}

SVDFeatureTasker::SVDFeatureTasker(map<string, string>& argv){
    if(argv.count("config"))
        config_file = string(argv["config"]);
    if(argv.count("train"))
        train_file = string(argv["train"]);
    if(argv.count("predict"))
        predict_file = string(argv["predict"]);
    if(argv.count("cross_validation"))
        cross_validation_file = string(argv["cross_validation"]);
    if(argv.count("user_feature"))
        user_feature_file = string(argv["user_feature"]);
    if(argv.count("item_feature"))
        item_feature_file = string(argv["user_feature"]);

    config = NULL;
    U = NULL;
    V = NULL;
    bias_factor = NULL;
    y_matrix = NULL;
}

void SVDFeatureTasker::TrainInit(){
    config = new Config(config_file);
    config->LoadConfig();

    y_matrix = new YMatrix(config);
    y_matrix->LoadData(train_file);

    U = new FactorMatrix(config, 'U');
    V = new FactorMatrix(config, 'V');
    bias_factor = new BiasFactor();

    U->Init();
    U->CheckNaNValue();
    V->Init();
    V->CheckNaNValue();
    bias_factor->Init(config, y_matrix->get_average());

    if(config->num_user_feature)
        U->LoadFeature(user_feature_file);

    if(config->num_item_feature)
        V->LoadFeature(item_feature_file);
}

void SVDFeatureTasker::PredictInit(const int& n_round){
    config = new Config(config_file);
    config->LoadConfig();

    // y_matrix = new YMatrix(config);
    // y_matrix->LoadData(train_name);

    U = new FactorMatrix(config, 'U');
    V = new FactorMatrix(config, 'V');
    bias_factor = new BiasFactor();

    U->Init();
    V->Init();
    bias_factor->Init(config, 0);

    stringstream ss;
    ss << n_round;
    string model_name = ss.str();
    string model_path = "./model/";
    FILE* fptr = fopen((model_path + model_name + ".model").c_str(), "rb");
    if(fptr == NULL){
        cerr << "Model '" << model_name << " was not existed "<<endl;
        exit(-1);
    }
    U->LoadMatrix(fptr);
    V->LoadMatrix(fptr);
    bias_factor->LoadBias(fptr);
}

void SVDFeatureTasker::Update(){
    U->Update(V, y_matrix, bias_factor);
    U->CheckNaNValue();
    V->Update(U, y_matrix, bias_factor);
    V->CheckNaNValue();
    bias_factor->BiasUpdate(U, V, y_matrix);

}

void SVDFeatureTasker::SaveModel(const int& round_num){
    cout << "Saving model for " << round_num << endl;
    stringstream ss;
    ss << round_num;
    string dump_model_name = ss.str();
    string model_path = "./model/";
    FILE* model_name = fopen((model_path + dump_model_name + ".model").c_str(), "wb");
    if(model_name == NULL){
        cerr << "Fail to open model file to write!" << endl;
        exit(-1);
    }

    U->SaveMatrix(model_name);
    V->SaveMatrix(model_name);
    bias_factor->SaveBias(model_name);

    fclose(model_name);
}

void SVDFeatureTasker::LoadModel(const int& round_num){
    stringstream ss;
    ss << round_num;
    string dump_model_name = ss.str();
    string model_path = "./model/";
    FILE* model_name = fopen((model_path + dump_model_name).c_str(), "rb");
    if(model_name == NULL){
        cerr << "Model file " << dump_model_name << " was not existed!" << endl;
        exit(-1);
    }

    U->LoadMatrix(model_name);
    V->LoadMatrix(model_name);
    bias_factor->LoadBias(model_name);

    fclose(model_name);
}

void SVDFeatureTasker::Train(const int& n_round){
    for(int round = 1; round <= n_round; round++){
        cout << "========================" << endl;
        cout << "Update for Round " << round << endl;
        Update();
        CalculateRMSE();
        CrossValidation();
        SaveModel(round);
    }
}

void SVDFeatureTasker::CalculateRMSE(){
    map<pair<int, int>, int> ratings = y_matrix->get_ratings();
    T_DATA u_matrix = U->get_matrix_val();
    T_DATA v_matrix = V->get_matrix_val();
    float* b_i = bias_factor->b_i;
    float* b_u = bias_factor->b_u;
    float average = bias_factor->average;

    float sum = 0.0;
    for(map<pair<int, int>, int>::iterator itr = ratings.begin();
            itr != ratings.end();
            itr++){
        int uid = (&(itr->first))->first;
        int iid = (&(itr->first))->second;
        int r = itr->second;
        float r_predict = 0.0;
        r_predict = ValuePredict(U, V, bias_factor, uid, iid);

        float err = r - r_predict;


        sum += err * err;
    }

    cout <<"RMSE:"<< sqrt(sum/ratings.size()) << endl;
}
void SVDFeatureTasker::CrossValidation(){
    ifstream fs;
    fs.open(cross_validation_file.c_str(), std::ios::in);
    if (fs.fail()){
        cerr << "Cross validation file" << cross_validation_file << "was not existed!" << endl;
        exit(-1);
    }

    string line;
    unsigned int line_num = 0;
    float sum = 0.0;
    while(getline(fs, line)){
        line_num ++;
        int uid, iid, rating;
        if(sscanf(line.c_str(), "%d\t%d\t%d", &uid, &iid, &rating) != 3){
            cerr << "Prdiction file format error at line " << line_num << endl;
            continue;
        }
        float y_ = ValuePredict(U, V, bias_factor, uid-1, iid-1);
        if(y_ - 1.0 < 0) y_ = 1.0;
        if(y_ - 5.0 > 0) y_ = 5.0;
        float err = rating - y_;
        sum += err * err;
    }
    printf("CrossValidationRMSE:%f\n", sqrt(sum/line_num));
}
void SVDFeatureTasker::Predict(){

    FILE* ans_ptr = fopen("pret.txt", "w");
    ifstream fs;
    fs.open(predict_file.c_str(), std::ios::in);
    if (fs.fail()){
        cerr << "Prediction file was not existed!" << endl;
        exit(-1);
    }

    string line;
    unsigned int line_num = 0;
    float sum = 0.0;
    while(getline(fs, line)){
        line_num ++;
        int uid, iid;
        if(sscanf(line.c_str(), "%d\t%d", &uid, &iid) != 2){
            cerr << "Prdiction file format error at line " << line_num << endl;
            continue;
        }
        float y_ = ValuePredict(U, V, bias_factor, uid-1, iid-1);
        fprintf(ans_ptr, "%f\n", y_);
    }

    fclose(ans_ptr);
}
