#ifndef __TYPES_H__
#define __TYPES_H__

#include <vector>

using namespace std;

typedef float T_DTYPE;
// typedef std::vector< std::vector<T_DTYPE> > T_MATRIX;
// typedef std::vector<T_DTYPE> T_VECTOR;

typedef T_DTYPE** T_MATRIX;
typedef T_DTYPE*  T_VECTOR;

typedef struct T_DATA_{
    T_MATRIX matrix;
    unsigned int num_row;
    unsigned int num_col;
}T_DATA;

#endif
