#include <iostream>
#include "tasker.h"

using namespace std;

// const string config_name = "data/config.conf";
// const string pred_name = "data/ratings.dat";
// const string test_pred_name = "data/ratings.dat.bak";
// const string user_feature = "data/user_feature.dat";
// const string user_feature_test = "data/user_feature.dat.bak";

map<string, string> my_argv;

int main(int argc, char** argv){
    int nround = 0;
    for(int i = 0; i < argc; i++){
        if(strcmp(argv[i], "-train") == 0){
            my_argv["train"] = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-config") == 0){
            my_argv["config"] = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-predict") == 0){
            my_argv["predict"] = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-nround") == 0){
            nround = atoi(argv[i+1]);
            i++;
        }
        else if(strcmp(argv[i], "-user_feature") == 0){
            my_argv["user_feature"] = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-cross_validation") == 0){
            my_argv["cross_validation"] = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-item_feature") == 0){
            my_argv["item_feature"] = argv[i+1];
            i++;
        }
    }

    SVDFeatureTasker tasker(my_argv);
    if(my_argv.count("train")){
        tasker.TrainInit();
        tasker.Train(nround);
    }
    else if(my_argv.count("predict")){
        tasker.PredictInit(nround);
        tasker.Predict();
    }
    else if(my_argv.count("cross_validation")){
        tasker.PredictInit(nround);
        tasker.CrossValidation();
    }
}
