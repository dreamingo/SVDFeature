#include "types.h"

typedef struct node_{
    // (which feature index split on the node)
    int m_featureNr;                
    // the prediction value(which feature value split on this node)
    T_DTYPE m_value;                   
    // pointer to node, if: feature[m_featureNr] <=  m_value
    struct node_* m_toSmallerEqual; 
    // pointer to node, if: feature[m_featureNr] > m_value
    struct node_* m_toLarger;       
    // a list of indices of the training samples in this node
    vector<int> m_trainSamples;
    // the length of m_trainSamples
    int m_nSamples;                 
} node;



typedef struct nodeReduced_{
    node* m_node;
    // m_size means the sample num in this node;
    // if this node was a leaf, m_size = 0
    unsigned int m_size;
} nodeReduced;

