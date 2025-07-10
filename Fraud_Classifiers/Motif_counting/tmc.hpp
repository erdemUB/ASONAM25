#ifndef tmc_hpp
#define tmc_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <queue>
#include <stack>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <string>
#include <initializer_list>

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <chrono>
#include <sys/stat.h>

using namespace std;

typedef chrono::duration<double> tms;
typedef long vertex;
typedef pair<vertex,bool> labeled_vertex;
typedef long timestamp;
typedef long amount;
typedef pair<timestamp,amount> edge_attr;
typedef pair<labeled_vertex, labeled_vertex> edge;
typedef pair<edge,bool> labeled_edge;
typedef pair<edge_attr, labeled_edge> event;//event.second.first.first.first = u and event.second.first.second.first = v, event.second.first is edge
typedef vector<event> key;  //a prefix or a motif
typedef pair<int, set<vertex>> counts;  //the count of the (prefix/motif) and the vertices in the (prefix/motif)
//typedef unordered_map<vector<event>, set<vertex>> prefix;
typedef unordered_map<vector<event>, pair<int, set<vertex>>> instancemap; //a hashtable of key and counts
typedef map<timestamp, set<labeled_edge>> eventmap;

template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std
{
    template<typename S, typename T> struct hash<pair<S, T>>
    {
        inline size_t operator()(const pair<S, T> & v) const
        {
            size_t seed = 0;
            ::hash_combine(seed, v.first);
            ::hash_combine(seed, v.second);
            return seed;
        }
    };
    
    template<typename S, typename T> struct hash<vector<pair<S, T>>>
    {
        inline size_t operator()(const vector<pair<S, T>> &k) const
        {
            size_t seed = 0;
            for (auto it=k.begin(); it != k.end(); ++it) {
                hash_combine(seed, it->first);
                hash_combine(seed, it->second);
            }
            return seed;
        }
    };
}

inline void print_time (FILE* fp, const string& str, tms t) {
    fprintf (fp, "%s %.6lf\n", str.c_str(), t.count());
    fflush(fp);
}

void createEvents (string filename, vector<event>& events, int bin); //Load and sort the event list
bool checkInduced(eventmap Emap, event e, vector<timestamp> T);
void countInstance (event e, instancemap& imap, set<vector<event>>& keys, int N_vtx, int N_event, int d_c, int d_w, string consecutive, string dgc, string amount_constraint, int a_c, int a_w, eventmap& Emap);    //Increment the instance count and update the prefix type
string encodeMotif(vector<event> instance); //identify the type of motif
void countMotif (event e, set<key>& pre, map<string, int>& motif_count, int N_vtx, int N_event, int d_c, int d_w, eventmap& Emap);
set<vertex> getNodes(vector<event> key);
void countSpecificmotif (event e, set<key>& pre, int& motif_count, string code_given, int N_vtx, int N_event, int d_c, int d_w);
char sconvert (int i);
#endif /* tmc_hpp */
