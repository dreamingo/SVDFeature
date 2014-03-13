#ifndef __GBDT_H__
#define __GBDT_H__

#include "types.h"
#include "tree.h"
#include <deque>
#include <map>
#include "Config.h"
#include "matrix.h"

class FactorMatrix;
class BiasFactor;
class YMatrix;

float ValuePredict(const FactorMatrix* U,
        const FactorMatrix* V,
        const BiasFactor* bias_factor,
        const int& uid,
        const int& iid);

class GBDT{
public:
    GBDT();
    GBDT(const Config* config);

    ~GBDT();

    bool Init(const Config* config);
    
    void PredictAllOutputs( const T_DATA& data, T_DTYPE* matrix_val_);

    bool ModelUpdate(const T_DATA& data,
            const FactorMatrix* U,
            const FactorMatrix* V,
            const BiasFactor* bias_factor,
            const char c_flag,
            const int k_dim,
            const YMatrix* y_matrix
            );

    void SaveWeights(FILE* fptr);
    void LoadWeights(FILE* ftpr);
private:
    void TrainSingleTree(
            node* largest_node, //The node with largest number of train sample
            deque< nodeReduced >& largest_nodes,
            const T_DATA& data,
            const FactorMatrix* U,
            const FactorMatrix* V,
            const int& k_dim,      //K_dim of the matrix to be update
            const char c_flag,     //a flag to indicate current matrix is 'U' or 'V'
            const YMatrix* y_matrix,
            const vector<int>& rand_feature_index,
            map<pair<int, int>, float >& g_ij,
            map<pair<int, int>, float >& h_ij,
            map<pair<int, int>, float >& y_predict
            );

    const Config* config;

    T_DTYPE PredictSingleTree(const node* n, const T_DATA& data, const int& dim);

    void CleanTree(node* n);

    void LoadTreeRecursive(node* n, FILE* fptr, string prefix);
    void SaveTreeRecursive(node* n, FILE* fptr);

private:
    // The root ptr of each tree
    node* m_trees_;
    // The max number of tree
    unsigned int m_max_epochs_;
    //The leaf number on a tree to prevent overfitting
    unsigned int m_max_tree_leafs_;
    //Randomly select part of feature for every single tree trainning
    //Stochastic gradient boosting
    unsigned int m_feature_subspace_size_;
    bool m_use_opt_splitpoint_;
    //shrinkage/learning rate
    double rt_lrate_;
    unsigned int m_train_epochs_;
    //Randomly select ratio of feature for every single tree trainning
    //Stochastic gradient boosting
    float m_data_sample_ratio_; 
    float rt_lambda;   //The regularization factor
};
#endif
