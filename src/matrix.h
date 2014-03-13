#ifndef __UMATRIX_H__
#define __UMATRIX_H__

#include "types.h"
#include "Config.h"
#include "gbdt.h"
#include <string>
#include <map>

using namespace std;

//Declaration
class GBDT;
class YMatrix;
class FactorMatrix;

//A class to manuplate the bias effect;
class BiasFactor{
public:
    BiasFactor();
    ~BiasFactor();
    void Init(const Config* config, const float& average);
    void SaveBias(FILE* fptr);
    void LoadBias(FILE* fptr);
    void BiasUpdate(const FactorMatrix* U,
            const FactorMatrix* V,
            const YMatrix* y_matrix);
public:
    //bias of item
    float* b_i;
    float* b_u;
    float average;

private:
    const Config* config;
    unsigned int num_user;
    unsigned int num_item;
};


class YMatrix{
public:
    YMatrix();
    YMatrix(const Config* config);
    ~YMatrix();
    //alloca space and set the parameters;
    void Init();
    bool LoadData(const string& file_name);
    //Use for debug;
    void PrintData(const string& file_name_out);
    float get_average() const;
    const map<pair<int, int>, int>& get_ratings() const;
    //store a user has rating on which items;
    const map<int, vector<int> >& get_user_rating_item_list() const;
    const map<int, vector<int> >& get_item_rated_by_user_list() const;
private:
    map<pair<int, int>, int>ratings;
    //store a user has rating on which items;
    map<int, vector<int> > user_rating_item_list;
    map<int, vector<int> > item_rated_by_user_list;
    const Config* config;
    unsigned int num_item_;
    unsigned int num_user_;
    // The number of the rating
    unsigned int num_rating_;
    float average;
    // a matrix with size num_user_ * num_item_;
    // int** rating_matrix_
    // The column index of the rating
    // int* col_index_;
    // // The row index of the rating
    // int* row_index_;
    // // The y_value of the rating;
    // int* y_value_;
};

class FactorMatrix{
public:
    FactorMatrix();
    FactorMatrix(const Config* config, const char c); ~FactorMatrix();
    void Update(const FactorMatrix* other_matrix, const YMatrix* y_matrix, BiasFactor* bias_factor);
    bool Init();
    bool LoadFeature(const string& file_in);
    void PrintData(const string& file_out);
    char get_c_flag() const ;
    int get_factor_num() const ;
    const T_DATA& get_matrix_val() const;
    void SaveMatrix(FILE* fptr);
    void LoadMatrix(FILE* fptr);
    void CheckNaNValue();
    void PrintMatrixCol(int col_num) const;

private:
    //Use for extra feature update
    const Config* config;
    //char c: a flag to represent this FactorMatrix was 
    //a user_latent_factor matrix or item_latent_factor_matrix;
    char c_flag;
    void GBDTUpdate(const FactorMatrix* U,
            const FactorMatrix* V,
            const BiasFactor* bias_factor,
            const YMatrix* y_matrix);
    //Use for no extra feature update
    void SGDUpdate(const FactorMatrix* this_matrix,
            const FactorMatrix* other_matrix,
            const YMatrix* y_matrix,
            BiasFactor* bias_factor);

    //The number of latent factor
    unsigned int factor_num_;
    //The number of the feature number for the gbdt
    unsigned int feature_num_;
    //The number of the item/user
    unsigned int num_;
    //A matrix with size factor_num_ * num to store the matrix val
    T_DATA matrix_val_;
    //A matrix with size num * feature_num to store the feature_val
    T_DATA feature_val_;
    /* U_k(i,x) = U_k(i,x) + lrate * f(i,x)*/
    GBDT* gbdt_for_each_factor_;
};

#endif
