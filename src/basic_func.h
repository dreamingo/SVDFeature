#ifndef __BASIC_FUNC_H__
#define __BASIC_FUNC_H__

#include <string>
#include <vector>
#include <time.h>
#include <memory.h>

using namespace std;

static void Strip(string& strVal){
    /*
     * Strip the white space both the left&right side of the string
     */
    unsigned int pos = 0;
    for(unsigned int i = 0; i < strVal.size(); i++){
        if(isspace((unsigned char)strVal[i])) ++pos;
        else break;
    }
    if(pos > 0) strVal.erase(0, pos);
    pos = 0;
    for(unsigned int i = 0; i < strVal.size(); i++){
        if(isspace((unsigned char)strVal[strVal.length() - i - 1])) ++ pos;
        else break;
    }
    if(pos > 0) strVal.erase(strVal.length() - pos);
}

static bool Split(const string& line, string& name, string& val){
    size_t nPos = -1;
    nPos = line.find("=", 0);
    if(nPos == std::string::npos || nPos == 0 || nPos == line.length() - 1) return false;
    else{
        name.assign(line, 0, nPos);
        val.assign(line, nPos+1, line.size() - nPos - 1);
        Strip(name);
        Strip(val);
    }
    return true;
}

static bool Split(const string& line, const char delimiter, vector<string>& featureVal){
    featureVal.clear();
    if (line.empty()) return false;
    size_t nPos = -1;
    size_t nPrePos = 0;
    string temp;
    while( (nPos = line.find(delimiter, nPrePos)) != std::string::npos ){
        if(nPos == 0) return false;
        temp.assign(line, nPrePos, nPos - nPrePos);
        Strip(temp);
        if(!temp.empty())
            featureVal.push_back(temp);
        nPrePos = nPos + 1;
    }
    temp.assign(line, nPrePos, line.length() - nPrePos);
    Strip(temp);
    if(!temp.empty())
        featureVal.push_back(temp);
    return true;
}


static void RandomSampleFromRange(vector<int>& arr, const int& num, const int&end_of_range, const bool& random_flag){
    if(random_flag){
        srand(time(0));
        bool* visited_id = new bool[end_of_range];
        memset(visited_id, 0, sizeof(visited_id));
        int n = 0;
        while(n != num){
            int id = rand()%end_of_range;
            if(!visited_id[id]){
                visited_id[id] = true;
                arr.push_back(id);
                n++;
            }
        }
        delete[] visited_id;
        visited_id = NULL;
    }
    else{
        for(int i = 0; i < num; i++)  arr.push_back(i);
    }
}
#endif
