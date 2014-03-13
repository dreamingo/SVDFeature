#include "matrix.h"
#include <fstream>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include "basic_func.h"

#define DEBUG

using namespace std;

BiasFactor::BiasFactor(){
    b_i = NULL;
    b_u = NULL;
    config = NULL;
    num_user = 0;
    num_item = 0;
}

BiasFactor::~BiasFactor(){
    delete[] b_i;
    delete[] b_u;
    b_i = NULL;
    b_u = NULL;
}

void BiasFactor::Init(const Config* config, const float& average){
    cout << "    Initilizing biasFactor...." << endl;
    srand48(time(NULL));
    this->config = config;
    this->num_user = config->num_user;
    this->num_item = config->num_item;
    this->average = average;

    b_i = new float[num_item]();
    for(int i = 0; i < num_item; i++)
        // b_i[i] = drand48() - 0.5;
        b_i[i] = 0;

    b_u = new float[num_user]();
    for(int i = 0; i < num_user; i++)
        // b_u[i] = drand48() - 0.5;
        b_u[i] = 0;

    cout << "    Finish Initilizing!" << endl;
}

void BiasFactor::SaveBias(FILE* fptr){
    fwrite( &(average), sizeof(average), 1, fptr );
    fwrite( &(num_user), sizeof(average), 1, fptr );
    fwrite( &(num_item), sizeof(average), 1, fptr );

    fwrite( b_i, sizeof(float), num_item, fptr );
    fwrite( b_u, sizeof(float), num_user, fptr );
}

void BiasFactor::LoadBias(FILE* fptr){
    fread( &(average), sizeof(average), 1, fptr );
    fread( &(num_user), sizeof(average), 1, fptr );
    fread( &(num_item), sizeof(average), 1, fptr );

    fread( b_i, sizeof(float), num_item, fptr );
    fread( b_u, sizeof(float), num_user, fptr );
}


void BiasFactor::BiasUpdate(const FactorMatrix* U,
        const FactorMatrix* V,
        const YMatrix* y_matrix){

    cout << "   Updating the bias factor..." << endl;

    map<pair<int, int>, int> ratings = y_matrix->get_ratings();
    T_DATA u_matrix = U->get_matrix_val();
    T_DATA v_matrix = V->get_matrix_val();

    for(map<pair<int, int>, int>::iterator itr = ratings.begin();
            itr != ratings.end();
            itr++){
        int uid = (&(itr->first))->first;
        int iid = (&(itr->first))->second;
        int r = itr->second;
        float r_predict = 0.0;
        r_predict = ValuePredict(U, V, this, uid, iid);

        // float err = r - r_predict - average 
        //     - b_u[uid] - b_i[iid];
        float err = r - r_predict;
        //Update for b_u
        b_u[uid] += config->sgd_lrate * (err - config->bias_reg * b_u[uid]);
        //Update for b_i
        b_i[iid] += config->sgd_lrate * (err - config->bias_reg * b_i[iid]);

        if(isnan(b_u[uid]) || isnan(b_i[iid])){
            cout << "........................" << endl;
            cout << "UID:" << uid << " iid:" << iid << endl;
            cout << "b_u:" << b_u[uid] << " b_i:" << b_i[iid] << endl;
            cout << "r_predict" << r_predict << endl;
            cout << "r:" << r << endl;
            exit(-1);
        }
        // cout << "err:" << err << endl;
        // cout << "sum:" << sum << endl;
    }
    cout << "   Finish Update BiaFactor!" << endl;
}


YMatrix::YMatrix(const Config* config){
    /*
     * Retrive data from config container;
     */
    this->config = config;
    average = 0.0;
    num_item_ = config->num_item;
    num_user_ = config->num_user;
    num_rating_ = config->num_rating;
    // rating_matrix_ = NULL;
    // col_index_ = NULL;
    // row_index_ = NULL;
    // y_value_ = NULL;
    Init();
}

void YMatrix::Init(){
    /*
     * Malloc space for the CSR array
     */
    // rating_matrix_ = new int*[num_user_];
    // for(int i = 0; i < rating_matrix_[num_user_]; i++){
    //     rating_matrix_[i] = new int[num_item_]();
    // }
    // col_index_ = new int[num_rating_];
    // row_index_ = new int[num_rating_];
    // y_value_ = new int[num_rating_];
    return;
}
YMatrix::YMatrix(){
    config = NULL;
    num_item_ = 0;
    num_user_ = 0;
    num_rating_ = 0;
    average = 0.0;
    // rating_matrix_ = NULL;
    // col_index_ = NULL;
    // row_index_ = NULL;
    // y_value_ = NULL;
}

YMatrix::~YMatrix(){
    /* Destruction */
    // for(int i = 0; i < rating_matrix_[num_user_]; i++)
    //     delete[] rating_matrix_[i];
    // delete[] rating_matrix_;
    // rating_matrix_ = NULL;
    // delete[] col_index_;
    // delete[] row_index_;
    // delete[] y_value_;
    // col_index_ = NULL;
    // row_index_ = NULL;
    // y_value_ = NULL;
    ratings.clear();
    user_rating_item_list.clear();
    item_rated_by_user_list.clear();
}

bool YMatrix::LoadData(const string& file_name){
    /*
     * Load data from file
     * The input format of the data
     * uid\t iid\t rating\n
     */
    ifstream fs;
    fs.open(file_name.c_str(), std::ios::in);
    if (fs.fail()){
        cerr << "Prediction file was not existed!" << endl;
        exit(-1);
    }
    string line;
    unsigned int line_num = 0;
    float sum = 0.0;
    while(getline(fs, line)){
        line_num ++;
        assert(line_num <= num_rating_);
        int rating = -1, uid = -1, iid = -1;
        if(sscanf(line.c_str(), "%d\t%d\t%d\t", &uid, &iid, &rating) != 3){
            cerr << "Prdiction file format error at line " << line_num << endl;
            continue;
        }

        assert(uid <= num_user_);
        assert(iid <= num_item_);

        sum += rating;
        ratings[make_pair<int, int>(uid-1, iid-1)] = rating;
        user_rating_item_list[uid-1].push_back(iid-1);
        item_rated_by_user_list[iid-1].push_back(uid-1);
        // rating_matrix_[uid - 1][iid - 1] = rating;
        // y_value_[line_num - 1] = rating;
        // row_index_[line_num - 1] = uid;
        // col_index_[line_num - 1] = iid;
    }
    average = sum / line_num;
}

float YMatrix::get_average() const{
    return this->average;
}
void YMatrix::PrintData(const string& file_name_out){
    /*
     * Use for debug, print Y_Matrix to the file_name_out;
     */ 
    // FILE* fptr = fopen(file_name_out.c_str(), "w");

    // assert(fptr != NULL);

    // for(unsigned int line = 0; line < num_rating_; line++){
    //     fprintf(fptr, "%d\t%d\t%d\n", y_value_[line], 
    //             row_index_[line], col_index_[line]);
    // }
    return;
}

const map<pair<int, int>, int>& YMatrix::get_ratings() const{
    return this->ratings;
}

const map<int, vector<int> >& YMatrix::get_user_rating_item_list() const{
    return this->user_rating_item_list;
}

const map<int, vector<int> >& YMatrix::get_item_rated_by_user_list() const{
    return this->item_rated_by_user_list;
}

FactorMatrix::FactorMatrix(){
    c_flag = 'U';
    factor_num_ = 32;
    feature_num_ = 0;
    num_ = 0;
    //Initialize for matrix_val_
    matrix_val_.num_row = 0;
    matrix_val_.num_col = 0;
    matrix_val_.matrix = NULL;

    //Initialize for feature_val_;
    feature_val_.num_row = 0;
    feature_val_.num_col = 0;
    feature_val_.matrix = NULL;
}

FactorMatrix::~FactorMatrix(){
    config = NULL;
    for(int i = 0; i < factor_num_; i++)
        delete[] matrix_val_.matrix[i];
    delete[] matrix_val_.matrix;

    for(int i = 0; i < num_; i++)
        delete[] feature_val_.matrix[i];
    delete[] feature_val_.matrix;
}

FactorMatrix::FactorMatrix(const Config* config, const char c){
    /*
     * Constructor for FactorMatrix;
     * @Parameter:
     *  config: A config container;
     *  char c: a flag to represent this FactorMatrix was 
     *  a user_latent_factor matrix or item_latent_factor_matrix;
     * @Returns:
     *  None;
     */
    this->c_flag = c;
    this->config = config;
    this->factor_num_ = config->num_factor;
    if ( c == 'U'){
        num_ = config->num_user;
        this->feature_num_ = config->num_user_feature;
    }
    else if( c == 'V'){
        num_ = config->num_item;
        this->feature_num_ = config->num_item_feature;
    }
    else assert(false);
}

bool FactorMatrix::Init(){
    /*
     * Init factorMatrix, allocspace for T_MATRIX matrix_val_ & feature_val
     */
    //matrix_val_ is a matrix of size factor_num_ * num_
    cout << "    Initilizing Matrix " << this->c_flag << " ...." << endl;

    srand48(time(NULL));
    matrix_val_.matrix = new T_DTYPE*[factor_num_];
    matrix_val_.num_row = factor_num_;
    matrix_val_.num_col = num_;
    for(unsigned int i = 0; i < factor_num_; i++){
        matrix_val_.matrix[i] = new T_DTYPE[num_]();
        for(int j = 0; j < num_; j++){
            matrix_val_.matrix[i][j] = drand48()-0.5;
        }
    }

    //feature_val is a matrix of size num_ * feature_num_;
    feature_val_.num_row = num_;
    feature_val_.num_col = feature_num_;
    if(feature_num_ != 0){
        feature_val_.matrix = new T_DTYPE*[num_];
        for(unsigned int i = 0; i < num_; i++)
            feature_val_.matrix[i] = new T_DTYPE[feature_num_]();
    }

    if(feature_num_ != 0){
        //If feature_num_!= 0, use GBDT to update the model
        //use a vecotr instead of a new operator to initilize,
        //which can use the GBDT non-default constructor;
        gbdt_for_each_factor_ = new GBDT[factor_num_]();
        for(int i = 0; i < factor_num_; i++)
            gbdt_for_each_factor_[i].Init(config);
    }

    cout << "    Finish Initilizing!" << endl;
    return true;
}

bool FactorMatrix::LoadFeature(const string& file_in){
    /*
     * FeatureFormat: uid/iid,fval1,fval2,fval3....
     */
    ifstream fs;
    fs.open(file_in.c_str(), std::ios::in);
    while(fs.fail()){
        cerr << "Fail to load featrue file!" << endl;
        exit(-1);
    }
    string line;
    unsigned int line_num = 0;
    while(getline(fs, line)){
        line_num++;

        assert(line_num <= num_);

        vector<string>val_list;
        if(!Split(line,',',val_list))
            cerr << "extra feature format wrong at line " << line_num << endl;
        //userId/Item id\t feat1\t feat2\t...
        assert(val_list.size() == feature_num_ + 1);

        for(int i = 1; i < val_list.size(); i++){
            //strtof str[beginptr:endptr] to float, if errno == ERANGE means the range of values
            //that can be represented is limited
            //if endptr == null means str[beginptr::]
            feature_val_.matrix[atoi(val_list[0].c_str())- 1][i-1] = (strtof(val_list[i].c_str(), NULL));
            if(errno == ERANGE)
                std::cerr << " feature out of the range: " << val_list[i] << std::endl;
        }
    }
}

char FactorMatrix::get_c_flag() const {
    return this->c_flag;
}

int FactorMatrix::get_factor_num() const{
    return this->factor_num_;
}

const T_DATA& FactorMatrix::get_matrix_val() const {
    return this->matrix_val_;
}

void FactorMatrix::PrintData(const string& file_out){
    //Use for debug;
    FILE* ptr = fopen(file_out.c_str(), "w");
    if(ptr == NULL){
        cerr << "test for FactorMatrix's fileout was not existed!" << endl;
        exit(-1);
    }
    for(int i = 0; i < num_; i++){
        fprintf(ptr,"%d,", i);
        for(int j = 0; j < feature_num_; j++){
            if(j != feature_num_- 1)
                fprintf(ptr,"%.2f,", feature_val_.matrix[i][j]);
            else
                fprintf(ptr,"%.2f\n", feature_val_.matrix[i][j]);
        }
    }
}

void FactorMatrix::SaveMatrix(FILE* fptr){
    //Store the private member of the matrix

    fwrite( &(c_flag), sizeof(char), 1, fptr );

    fwrite( &(factor_num_), sizeof(unsigned int), 1, fptr );

    fwrite( &(feature_num_), sizeof(unsigned int), 1, fptr );

    fwrite( &(num_), sizeof(int), 1, fptr );

    //Store the T_DATA matrix_val
    fwrite( &(matrix_val_.num_row), sizeof(unsigned int), 1, fptr);
    fwrite( &(matrix_val_.num_col), sizeof(unsigned int), 1, fptr);

    if(matrix_val_.num_row != 0 && matrix_val_.num_col != 0){
        for(int i = 0; i < matrix_val_.num_row; i++)
            fwrite( matrix_val_.matrix[i], sizeof(T_DTYPE), 
                    matrix_val_.num_col, fptr);
    }

    //Store the T_DATA feature_val_
    fwrite( &(feature_val_.num_row), sizeof(unsigned int), 1, fptr);
    fwrite( &(feature_val_.num_col), sizeof(unsigned int), 1, fptr);

    if(feature_val_.num_row != 0 && feature_val_.num_col != 0){
        for(int i = 0; i < feature_val_.num_row; i++)
            fwrite( feature_val_.matrix[i], sizeof(T_DTYPE), 
                    feature_val_.num_col, fptr);
    }

    //The model has extra feature
    if(feature_num_ != 0){
        //Each feactor has a gbdt
        for(int i = 0; i < factor_num_; i++){
            gbdt_for_each_factor_[i].SaveWeights(fptr);
        }
    }
}

void FactorMatrix::LoadMatrix(FILE* fptr){
    /*
     * Load matrix from a model file
     */
    fread( &(c_flag), sizeof(char), 1, fptr );

    fread( &(factor_num_), sizeof(unsigned int), 1, fptr );

    fread( &(feature_num_), sizeof(unsigned int), 1, fptr );

    fread( &(num_), sizeof(int), 1, fptr );

    //Store the T_DATA matrix_val
    fread( &(matrix_val_.num_row), sizeof(unsigned int), 1, fptr);
    fread( &(matrix_val_.num_col), sizeof(unsigned int), 1, fptr);

    if(matrix_val_.num_row != 0 && matrix_val_.num_col != 0){
        for(int i = 0; i < matrix_val_.num_row; i++)
            fread( matrix_val_.matrix[i], sizeof(T_DTYPE), 
                    matrix_val_.num_col, fptr);
    }

    //Store the T_DATA feature_val_
    fread( &(feature_val_.num_row), sizeof(unsigned int), 1, fptr);
    fread( &(feature_val_.num_col), sizeof(unsigned int), 1, fptr);

    if(feature_val_.num_row != 0 && feature_val_.num_col != 0){
        for(int i = 0; i < feature_val_.num_row; i++)
            fread( feature_val_.matrix[i], sizeof(T_DTYPE), 
                    feature_val_.num_col, fptr);
    }

    //The model has extra feature
    if(feature_num_ != 0){
        //Each feactor has a gbdt
        for(int i = 0; i < factor_num_; i++){
            gbdt_for_each_factor_[i].LoadWeights(fptr);
        }
    }
}

void FactorMatrix::Update(const FactorMatrix* other_matrix, 
        const YMatrix* y_matrix,
        BiasFactor* bias_factor){

    if(feature_num_ == 0){
        cout << "    SGD Updating the Matrix "<< c_flag << "... " << endl;
        SGDUpdate(this, other_matrix, y_matrix, bias_factor);
        cout << "    Finish SGD Update!" << endl;
    }
    else{
        if(c_flag == 'U'){
            cout << "    Updating the GBDT of Matrix U ..." << endl;
            GBDTUpdate(this, other_matrix, bias_factor, y_matrix);
        }
        else{
            cout << "    Updating the GBDT of Matrix V ..." << endl;
            GBDTUpdate(other_matrix,this, bias_factor, y_matrix);
        }
    }
}

void FactorMatrix::GBDTUpdate(const FactorMatrix* U,
        const FactorMatrix* V,
        const BiasFactor* bias_factor,
        const YMatrix* y_matrix)
{
    for(int i = 0; i < factor_num_; i++){
        printf("    Updating the %dth factor GBDT of matrix %c ...\n", i, c_flag);
        this->gbdt_for_each_factor_[i].ModelUpdate(this->feature_val_, 
                U,V, bias_factor, this->c_flag, i, y_matrix);
    }
}

void FactorMatrix::CheckNaNValue(){
    for(int i = 0; i < factor_num_; i++){
        for(int j = 0; j < num_; j++){
            if(isnan(matrix_val_.matrix[i][j])){
                printf("NaN occur in Matrix_%c[%d][%d]!\n", c_flag, i, j);
            }
        }
    }
}
void FactorMatrix::SGDUpdate(const FactorMatrix* this_matrix, 
        const FactorMatrix* other_matrix,
        const YMatrix* y_matrix,
        BiasFactor* bias_factor
        ){
    map<pair<int, int>, int> ratings = y_matrix->get_ratings();
    const T_DATA& other_matrix_val = other_matrix->get_matrix_val();

    for(map<pair<int, int>, int>::iterator itr = ratings.begin();
            itr != ratings.end();
            itr++){
        int uid = (&(itr->first))->first;
        int iid = (&(itr->first))->second;
        int r = itr->second;

        float r_predict = 0.0;
        if(this->c_flag == 'U')
            r_predict = ValuePredict(this_matrix, other_matrix, bias_factor, uid, iid);
        else
            r_predict = ValuePredict(other_matrix, this_matrix, bias_factor, uid, iid);

        // float err = r - r_predict - bias_factor->average 
        //     - bias_factor->b_u[uid] - bias_factor->b_i[iid];
        float err = r - r_predict;

        for(int i = 0; i < factor_num_; i++){
            if(this->c_flag == 'U'){
                matrix_val_.matrix[i][uid] += config->sgd_lrate * 
                    (err * other_matrix_val.matrix[i][iid] - 
                     config->sgd_lambda * matrix_val_.matrix[i][uid]);

                if(isnan(matrix_val_.matrix[i][uid])){
                    printf("error occur in updating matrix %c, uid:%d iid:%d\n", this->c_flag, uid, iid);
                    exit(-1);
                }

            }
            else{
                matrix_val_.matrix[i][iid] += config->sgd_lrate * 
                    (err * other_matrix_val.matrix[i][uid] - config->sgd_lambda * matrix_val_.matrix[i][iid]);
                if(isnan(matrix_val_.matrix[i][iid])){
                    printf("error occur in updating matrix %c, uid:%d iid:%d\n", this->c_flag, uid, iid);
                    exit(-1);
                }
            }
        }
    }
}


void FactorMatrix::PrintMatrixCol(int col_num) const{
    //print the laten factor of a user/item with id 'col_num'
    printf("The laten factor of %c with id:%d\n", c_flag, col_num);
    for(int i = 0 ; i < factor_num_; i++){
        printf("%c[%d][%d]:%f ", c_flag, i, col_num, matrix_val_.matrix[i][col_num]);
    }
    printf("\n");
}
