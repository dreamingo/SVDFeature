#include "gbdt.h"
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <set>
#include <algorithm>
#include "basic_func.h"

using namespace std;

float ValuePredict(const FactorMatrix* U,
        const FactorMatrix* V,
        const BiasFactor* bias_factor,
        const int& uid,
        const int& iid){

    float y_ = 0.0;
    T_DATA U_matrix_val = U->get_matrix_val();
    T_DATA V_matrix_val = V->get_matrix_val();
    for(int i = 0; i < U->get_factor_num(); i++){
        y_ += U_matrix_val.matrix[i][uid] * V_matrix_val.matrix[i][iid];
    }

    y_ += bias_factor->average + bias_factor->b_i[iid] + bias_factor->b_u[uid];
    if(isnan(y_)){
        cout << "Occur error in valuePredict function!" << endl;
        printf("uid:%d\t iid%d\n", uid, iid);
        for(int i = 0; i < U->get_factor_num(); i++){
            printf("U[%d]:%f\t V[%d]:%f\n", i, U_matrix_val.matrix[i][uid], i,
                    V_matrix_val.matrix[i][iid]);
        }
        exit(-1);
    }
    return y_;
}
static bool compareNodeReduced ( nodeReduced n0, nodeReduced n1 ){
    return n0.m_size < n1.m_size;
}


static int64_t Milliseconds(){
    /*
     * Return a curTime with milli seconds
     */
    // struct timeval {
    //   time_t      tv_sec;     /* seconds */
    //   suseconds_t tv_usec;    /* microseconds */
    // };
    // microsecond = 1/1000000 sec;微秒
    // miliseconds = 1/1000 sec   毫秒 
    struct timeval t;    
    ::gettimeofday(&t, NULL);    
    int64_t curTime;    
    curTime = t.tv_sec;    
    curTime *= 1000;              // sec -> msec    
    curTime += t.tv_usec / 1000;  // usec -> msec    
    return curTime;
}

GBDT::GBDT(){
    this->config = NULL;
    m_trees_ = NULL;
    m_max_epochs_ = 400;
    m_max_tree_leafs_ = 20;
    m_feature_subspace_size_ = 20;
    m_use_opt_splitpoint_ = true;
    rt_lrate_ = 0.01;
    m_train_epochs_ = 0;
    m_data_sample_ratio_ = 1.0;
    rt_lambda = 0.0;
}

bool GBDT::Init(const Config* config){
    this->config = config;
    m_trees_ = NULL;
    m_max_epochs_ = config->m_max_epochs;
    m_max_tree_leafs_ = config->m_max_tree_leafs;
    m_feature_subspace_size_ = config->m_feature_subspace_size;
    m_use_opt_splitpoint_ = true;
    rt_lrate_ = config->rt_lrate;
    m_train_epochs_ = 0;
    m_data_sample_ratio_ = config->m_data_sample_ratio;
    rt_lambda = config->rt_lambda;

    m_trees_ = new node[m_max_epochs_];
    for(int i = 0; i < m_max_epochs_; i++){
        m_trees_[i].m_featureNr = -1;
        m_trees_[i].m_value = 1e10;
        m_trees_[i].m_toSmallerEqual = NULL;
        m_trees_[i].m_toLarger = NULL;
        // m_trees_[i].m_trainSamples = NULL;
        m_trees_[i].m_nSamples = -1;
    }
#ifdef DEBUG
    cout << "-------configure--------" << endl;
    cout <<  "  max_epochs: " << m_max_epochs << endl;
    cout <<  "  max_tree_leafes: " << m_max_tree_leafes << endl;
    cout <<  "  feature_subspace_size: " << m_feature_subspace_size_ << endl;
    cout <<  "  use_opt_splitpoint: " << m_use_opt_splitpoint << endl;
    cout <<  "  learn_rate: " << rt_lrate_ << endl;
    cout <<  "  data_sample_ratio: " << m_data_sample_ratio << endl;
    cout << endl;
#endif
}

GBDT::~GBDT(){
    delete[] m_trees_;
    m_trees_ = NULL;
}

bool GBDT::ModelUpdate(const T_DATA& data, 
        const FactorMatrix* U,
        const FactorMatrix* V,
        const BiasFactor* bias_factor,
        const char c_flag,   //The gbdt of 'U' or 'V'
        const int k_dim,     //The k_dim of the matrix to be update;
        const YMatrix* y_matrix){
    /*
     *@Procedure:
     *  2. If data_sample_ratio < 1, sample the data randomly
     *  3. if m_feature_subspace_size < m_dim, sample the feature randomly 
     *  4. for  j=0; j<m_max_tree_leafes; j++  : TrainSingleTree();
     *  5. clean the tree since some part of it was not necessary for prediction
     *@Param:
data: the extra featrue matrix with size num_user/item * num_feature
factor_matrix, the oposite factor_matrix(U/V)
y_matrix, the rating matrix
     *
     */
    if(m_train_epochs_ > m_max_epochs_) return false;
    unsigned int n_sample_num = data.num_row;
    unsigned int n_feature_num = data.num_col;

    //=== Sample the data, stochastic gradient boosting === ;
    //Use only a part of sample for a tree
    //just some like bagging
    bool random_flag = true;
    int train_sample_num = int(n_sample_num * m_data_sample_ratio_);
    if (train_sample_num < 10) train_sample_num = n_sample_num;
    if (train_sample_num == n_sample_num) random_flag = false;

    m_trees_[m_train_epochs_].m_nSamples = train_sample_num;

    // RandomSampleFromRange(int* arr, num, end_of_range, random_flag);
    RandomSampleFromRange(m_trees_[m_train_epochs_].m_trainSamples, train_sample_num, n_sample_num, random_flag);

    //=== Sample the feature === ;
    random_flag = true;
    assert(m_feature_subspace_size_ <= n_feature_num);
    vector<int> rand_feature_index; 

    if(m_feature_subspace_size_ == n_feature_num) random_flag = false;

    RandomSampleFromRange(rand_feature_index, m_feature_subspace_size_,
            n_feature_num, random_flag);

    //Each time we select the node with most sample to split;
    //here m_size is the train_sample_num in one node;
    //if this node was a leaf, m_size = 0
    //----init first node for split----
    deque< nodeReduced > largest_nodes;
    nodeReduced first_node;
    first_node.m_node = & ( m_trees_[m_train_epochs_] );
    // Then size of the trainning sample in this node
    first_node.m_size = train_sample_num;
    // init m_node to be the root of the update tree in this round
    largest_nodes.push_back ( first_node );
    //heap for select largest num node
    push_heap( largest_nodes.begin(), largest_nodes.end(), compareNodeReduced );  
    //g_ij = Y_ij - Y_pred_ij;
    //h_ij = -Y_pred_ij;
    map<pair<int, int>, int> ratings = y_matrix->get_ratings();;
    //store a user has rating on which items;
    map<int, vector<int> > user_rating_item_list = y_matrix->get_user_rating_item_list();
    map<int, vector<int> > item_rated_by_user_list = y_matrix->get_item_rated_by_user_list();

    map<pair<int, int>, float > g_ij;
    map<pair<int, int>, float > h_ij;
    map<pair<int, int>, float > y_predict;
    if(c_flag == 'U'){
        for(int i = 0; i < train_sample_num; i++){
            int uid = first_node.m_node->m_trainSamples[i];
            for(int j = 0; j < user_rating_item_list[uid].size(); j++){
                int iid = user_rating_item_list[uid][j];
                float y_ = ValuePredict(U, V, bias_factor, uid, iid);
                pair<int, int>p = make_pair<int, int>(uid, iid);
                y_predict[p] = y_;
                g_ij[p] = ratings[p] - y_;
                h_ij[p] = -1 * y_;
            }
        }
    }
    else{
        for(int i = 0; i < train_sample_num; i++){
            int iid = first_node.m_node->m_trainSamples[i];
            for(int j = 0; j < item_rated_by_user_list[iid].size(); j++){
                int uid = item_rated_by_user_list[iid][j];
                float y_ = ValuePredict(U, V, bias_factor, uid, iid);
                pair<int, int> p = make_pair<int, int>(uid, iid);
                y_predict[p] = y_;
                g_ij[p] = ratings[p] - y_;
                h_ij[p] = -1 * y_;
            }
        }
    }

    //--Build the tree
    for(int i = 0; i < m_max_tree_leafs_; i++){
        node* largest_node = largest_nodes[0].m_node;
        TrainSingleTree(
                largest_node, //The node with largest number of train sample
                largest_nodes,//A list contain for the nodeReduced
                data,
                U,
                V,
                k_dim,      //K_dim of the matrix to be update
                c_flag,     //a flag to indicate current matrix is 'U' or 'V'
                y_matrix,
                rand_feature_index,
                g_ij, h_ij, y_predict
                );
    }
    CleanTree( &(m_trees_[m_train_epochs_]) );
    //Finish trainning a tree
    m_train_epochs_ ++;

    rand_feature_index.clear();
    return true;
}

void GBDT::TrainSingleTree(
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
        )
{

    //--- break criteria: tree size limit or too less training samples
    unsigned int nS = largest_nodes.size(); //the tree leafes
    if ( nS >= m_max_tree_leafs_ || largest_node->m_nSamples <= 1 )
        return;

    vector<int>& rand_sample_index = largest_node->m_trainSamples;

    //-- delete the current node (is currently the largest element in the heap)
    if ( largest_nodes.size() > 0 ){
        //largestNodes.pop_front();
        // pop_heap move the root of the heap to the end of the container and pop_back
        // remove it out of the container 
        // pop it when the node was not a leaf
        pop_heap ( largest_nodes.begin(),largest_nodes.end(),compareNodeReduced );
        largest_nodes.pop_back();
    }

    float gain = 0.0, G_all = 0.0, H_all = 0.0;
    //-- Initialized the instance set;
    set<pair<int, int> > instance_set;
    map<pair<int, int>, int> ratings = y_matrix->get_ratings();;
    //store a user has rating on which items;
    map<int, vector<int> > user_rating_item_list = y_matrix->get_user_rating_item_list();
    map<int, vector<int> > item_rated_by_user_list = y_matrix->get_item_rated_by_user_list();

    // for(map<int, vector<int> >::iterator itr = user_rating_item_list.begin();
    //         itr != user_rating_item_list.end();
    //         itr++){
    //     printf("----------\nuser[%d]:", itr->first);
    //     for(int i = 0; i < itr->second.size(); i++){
    //         printf("%d ", itr->second[i]);
    //     }
    //     printf("\n");
    // }
    if(c_flag == 'U'){
        for(int i = 0; i < rand_sample_index.size(); i++){
            int uid = rand_sample_index[i];
            assert(uid < config->num_user);
            for(int j = 0; j < user_rating_item_list[uid].size(); j++){
                int iid = user_rating_item_list[uid][j];
                assert(iid < config->num_item);
                instance_set.insert(make_pair<int, int>(uid, iid));
            }
        }
    }
    else{
        for(int i = 0; i < rand_sample_index.size(); i++){
            int iid = rand_sample_index[i];
            assert(iid < config->num_item);
            for(int j = 0; j < item_rated_by_user_list[iid].size(); j++){
                int uid = item_rated_by_user_list[iid][j];
                assert(uid < config->num_user);
                instance_set.insert(make_pair<int, int>(uid, iid));
            }
        }
    }

    // -- Initializa the G_all & H_all;
    for(set<pair<int, int> >::iterator itr = instance_set.begin();
            itr != instance_set.end();
            itr++){
        int uid = (itr)->first;
        int iid = (itr)->second;
        float val = 0.0;
        if(c_flag == 'U')
            val = V->get_matrix_val().matrix[k_dim][iid];
        else if(c_flag == 'V')
            val = U->get_matrix_val().matrix[k_dim][uid];

        G_all += g_ij[make_pair<int, int>(uid, iid)] * val;
        H_all += h_ij[make_pair<int, int>(uid, iid)] * val * val;
    }

    // ======================================
    // Choose the feature with its split value, 
    // which can reduce the loss function most
    int best_feature_index = -1;
    float best_split_val_all = 0;

    for(int f = 0; f < rand_feature_index.size(); f++){
        int f_index = rand_feature_index[f];
        float best_gain_in_one_feature = 0;
        float best_split_val = 0;

        vector<pair<float, pair<int,int> > > sorted_feature;

        for(set<pair<int, int> >::iterator itr = instance_set.begin();
                itr != instance_set.end();
                itr++){
            pair<int, int> cord = *itr;
            int uid = (itr)->first;
            int iid = (itr)->second;
            if(c_flag == 'U'){
                sorted_feature.push_back(make_pair<float, pair<int, int> >
                        (data.matrix[uid][f_index], cord));
            }
            else{
                sorted_feature.push_back(make_pair<float, pair<int, int> >
                        (data.matrix[iid][f_index], cord));
            }
        }
        sort(sorted_feature.begin(), sorted_feature.end());

        float G_left = 0.0, G_right = 0.0;
        float H_left = 0.0, H_right = 0.0;

        for(int i = 0; i < sorted_feature.size(); i++){
            float f_val = sorted_feature[i].first;
            int uid = sorted_feature[i].second.first;
            int iid = sorted_feature[i].second.second;

            float val;
            if(c_flag == 'U')
                val = V->get_matrix_val().matrix[k_dim][iid];
            else
                val = U->get_matrix_val().matrix[k_dim][uid];

            // G_left += g_ij[uid][iid] * val;
            G_left += g_ij[make_pair<int, int>(uid,iid)] * val;
            H_left += h_ij[make_pair<int, int>(uid,iid)] * val * val;

            // printf("G_left:%f, G_all:%f\n", G_left, G_all);
            // printf("H_left:%f, H_all:%f\n", H_left, H_all);

            // assert(G_left <= G_all);
            // assert(H_left <= H_all);

            G_right = G_all - G_left;
            H_right = G_all - H_left;

            float loss_reduction = (G_left * G_left)/(H_left + rt_lambda) + 
                (G_right * G_right)/(H_right + rt_lambda) - 
                (G_all * G_all)/(H_all + rt_lambda);
            if( loss_reduction > best_gain_in_one_feature){
                best_gain_in_one_feature = loss_reduction;
                best_split_val = f_val;
            }
        }

        if(best_gain_in_one_feature > gain){
            gain = best_gain_in_one_feature;
            best_feature_index = f_index;
            best_split_val_all = best_split_val;
        }
    }

    largest_node->m_featureNr = best_feature_index;
    largest_node->m_value = best_split_val_all;

    node* left_node = new node();
    node* right_node = new node();
    float sum_left = 0.0, sum_right = 0.0;

    for(int i = 0; i < rand_sample_index.size(); i++){
        int id = rand_sample_index[i];
        if(data.matrix[id][best_feature_index] <= best_split_val_all){
            left_node->m_trainSamples.push_back(id);
            sum_left += data.matrix[id][best_feature_index];
        }
        else{
            right_node->m_trainSamples.push_back(id);
            sum_right += data.matrix[id][best_feature_index];
        }
    }

    float mean_right = sum_right/right_node->m_trainSamples.size() ;
    float mean_left = sum_left/left_node->m_trainSamples.size() ;

    //Break if too less sample
    //to be a leaf in the tree;
    if(left_node->m_trainSamples.size() < 1 ||
            right_node->m_trainSamples.size() < 1)
    {
        largest_node->m_featureNr = -1;
        largest_node->m_value = left_node->m_trainSamples.size() < 1 ?
            mean_right:
            mean_left;
        largest_node->m_toSmallerEqual = NULL;
        largest_node->m_toLarger = NULL;
        largest_node->m_trainSamples.clear();
        largest_node->m_nSamples = 0;

        nodeReduced current_node;
        current_node.m_node = largest_node;
        current_node.m_size = 0;

        largest_nodes.push_back ( current_node );
        push_heap ( largest_nodes.begin(), largest_nodes.end(), 
                compareNodeReduced );

        left_node->m_trainSamples.clear();
        right_node->m_trainSamples.clear();
        delete left_node;
        delete right_node;
    }
    else{
        left_node->m_nSamples = left_node->m_trainSamples.size();
        left_node->m_featureNr = -1;
        left_node->m_toSmallerEqual = NULL;
        left_node->m_toLarger = NULL;
        left_node->m_value = mean_left;
        largest_node->m_toSmallerEqual = left_node;

        right_node->m_nSamples = right_node->m_trainSamples.size();
        right_node->m_featureNr = -1;
        right_node->m_toSmallerEqual = NULL;
        right_node->m_toLarger = NULL;
        right_node->m_value = mean_right;
        largest_node->m_toLarger = right_node;

        // add the new two nodes to the heap
        nodeReduced lowNode, hiNode;
        lowNode.m_node = left_node;
        lowNode.m_size = left_node->m_nSamples;
        hiNode.m_node = right_node;
        hiNode.m_size = right_node->m_nSamples;

        largest_nodes.push_back ( lowNode );
        push_heap ( largest_nodes.begin(), largest_nodes.end(), compareNodeReduced );

        largest_nodes.push_back ( hiNode );
        push_heap ( largest_nodes.begin(), largest_nodes.end(), compareNodeReduced );
    }

    g_ij.clear();
    h_ij.clear();
    y_predict.clear();
}

void GBDT::CleanTree(node* n){
    /*Clear the train_Sample in the tree;
     * @Paramter:
     *  node* n; the node in the tree;
     */
    if(n->m_nSamples != 0){
        n->m_trainSamples.clear();
        n->m_nSamples = 0;
    }
    if(n->m_toSmallerEqual)
        CleanTree(n->m_toSmallerEqual);
    if(n->m_toLarger)
        CleanTree(n->m_toLarger);
}

void GBDT::PredictAllOutputs(const T_DATA& data, T_DTYPE*  k_dim_matrix_val_){
    /*
     * @Param:
     *  data: an extra feature matrix with num * feature_size;
     *  k_dim_matrix_val_: The kth dim of U/V to be update
     */
    unsigned int n_sample = data.num_row;
    for( unsigned int i = 0; i < n_sample; i++){
        double sum = 0.0;
        for(unsigned int j = 0; j < m_train_epochs_; j++){
            float v = PredictSingleTree( &(m_trees_[j]), data, i );
            sum += v * rt_lrate_;
        }
        k_dim_matrix_val_[i] = sum;
    }
}

T_DTYPE GBDT::PredictSingleTree(const node* n, const T_DATA& data, const int& dim){
    /*
     * @Param:
     *  node* n: The node in the Single Tree
     *  T_DATA& data, an extra feature matrix with num * feature_size;
     *  int dim:    The  dim_th user/item index
     * @Return:
     * The prediction value of this single tree;
     */
    int n_features = data.num_col;
    int nr = n->m_featureNr;
    if(nr < -1 || nr >= n_features)
    {
        cerr << "Feature nr:" << nr << endl;
        assert(false);
    }
    if( n->m_toSmallerEqual == 0 && n->m_toLarger == 0)
        return n->m_value;

    if(data.matrix[dim][nr] <= n->m_value)
        return PredictSingleTree(n->m_toSmallerEqual, data, dim);
    else
        return PredictSingleTree(n->m_toLarger, data, dim);
}


void GBDT::SaveWeights(FILE* fptr){

    // save learnrate
    fwrite(&rt_lrate_, sizeof(rt_lrate_), 1, fptr);

    // save number of epochs
    fwrite (&m_train_epochs_, sizeof ( m_train_epochs_ ), 1, fptr );

    // save trees
    for ( unsigned int j=0;j<m_train_epochs_+1;j++ )
        SaveTreeRecursive ( & ( m_trees_[j] ), fptr );
}

void GBDT::SaveTreeRecursive ( node* n, FILE* fptr )
{
    //cout << "debug_save: " << n->m_value << " " << n->m_featureNr << endl;
    fwrite ( n, sizeof ( node ), 1, fptr);
    if ( n->m_toSmallerEqual )
        SaveTreeRecursive ( n->m_toSmallerEqual, fptr );
    if ( n->m_toLarger )
        SaveTreeRecursive ( n->m_toLarger, fptr );
}

void GBDT::LoadWeights(FILE* fptr)
{
    // load learnrate
    fread ( &rt_lrate_, sizeof ( rt_lrate_ ), 1, fptr );

    // load number of epochs
    fread ( &m_train_epochs_, sizeof ( m_train_epochs_ ), 1, fptr );


    // allocate and load the trees
    m_trees_ = new node[m_train_epochs_+1];
    for ( unsigned int j=0;j<m_train_epochs_+1;j++ )
    {
        std::string prefix = "";
        LoadTreeRecursive ( & ( m_trees_[j] ), fptr, prefix );
    }
}

void GBDT::LoadTreeRecursive ( node* n, FILE* fptr, std::string prefix ){
    fread ( n, sizeof ( node ), 1, fptr );

    //cout << prefix;
    //cout << "debug_load: " << n->m_value << " " << n->m_featureNr << endl;
    if ( n->m_toLarger == 0 && n->m_toSmallerEqual == 0 ){
        assert( n->m_featureNr == -1);
    }
    prefix += "    ";
    if ( n->m_toSmallerEqual ){
        n->m_toSmallerEqual = new node;
        LoadTreeRecursive ( n->m_toSmallerEqual, fptr , prefix);
    }
    if ( n->m_toLarger ){
        n->m_toLarger = new node;
        LoadTreeRecursive ( n->m_toLarger, fptr , prefix);
    }
}

