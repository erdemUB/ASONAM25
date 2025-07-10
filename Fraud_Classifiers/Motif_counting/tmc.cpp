#include "tmc.hpp"

void createEvents (string filename, vector<event>& events, int bin){
    ifstream in(filename);
    string line;
    
    while (getline(in, line)) {
        //cout << "Starts here" << endl;
        if (line[0] != '%' && line[0] != '#'){
            stringstream ss (line);
            vertex u, v;
            labeled_vertex labeled_u, labeled_v;
            bool u_label, v_label, e_label;
            timestamp t;
            amount a;
            edge_attr ta;
            edge e;
            labeled_edge labeled_e;
            ss >> u >> v >> t >> a >> u_label >> v_label >> e_label;
            // if (u==1926){
            // cout << "u: " << u << " v: " << v << " t: " << t << " a: "<<a  <<" u_label: " << u_label << " v_label: " << v_label << endl;
            // }
//            int x{60};
            t = t/bin*bin;
            //
            if (u != v) {
                labeled_u = make_pair(u,u_label);
                labeled_v = make_pair(v,v_label);
                e = make_pair(labeled_u, labeled_v);
                labeled_e = make_pair(e,e_label);
                ta = make_pair(t,a);
                events.push_back(make_pair(ta, labeled_e));
            }
        }
    }
    sort(events.begin(), events.end());
//    events.erase(unique(events.begin(), events.end()),events.end()); //remove duplicates
    return;
}

void countInstance (event e, instancemap& imap, set<vector<event>>& keys, int N_vtx, int N_event, int d_c, int d_w,  string consecutive, string dgc, string amount_constraint, int a_c, int a_w, eventmap& Emap){
    labeled_vertex labeled_u = e.second.first.first;
    labeled_vertex labeled_v = e.second.first.second;
    vertex u = e.second.first.first.first;
    vertex v = e.second.first.second.first;
    vector<vector<event>> new_motif;    //used to store the new motifs
    for (auto it = keys.begin(); it != keys.end();) {   //for each current prefix
        vector<event> key = *it;
        if ((e.first.first - key.front().first.first <= d_w && e.first.first - key.back().first.first <= d_c) && ((amount_constraint == "YES"  && e.first.second - key.back().first.second <= a_c) || (amount_constraint == "NO")) ){  //check delta C and delta W
            // cout<<e.first.second<<" " <<key.front().first.second<< " " <<key.back().first.second<<endl;
            // && abs(e.first.second - key.front().first.second) <= a_w
            if (key.size() < N_event) { //check the number of events
                set<vertex> nodes = imap[key].second;
                nodes.insert(u);
                nodes.insert(v);
                if (nodes.size() <= N_vtx) {    //check the number of vertices
            
                    if (imap[key].second.find(u)!=imap[key].second.end() || imap[key].second.find(v)!=imap[key].second.end()) {
                        if (key.back().first.first!=e.first.first) { //check synchronous events
                            vector<event> motif = key;
                            vector<timestamp> T;
                            for (int i=0; i<motif.size(); i++) {
                                if (motif[i].second.first.first.first!=u||motif[i].second.first.second.first!=v) {
                                    T.push_back(motif[i].first.first);
                                }
                            }
                            if (checkInduced(Emap, e, T)||dgc=="NO") {
                                motif.push_back(e);
                                new_motif.push_back(motif);
                                imap[motif].first += imap[key].first;
                                imap[motif].second = nodes;
                            }
                        }
                    }

                    
                }
                if (consecutive == "YES") {
                    it = keys.erase(it);
                    continue;
                }
                ++it;
            } else {
                it = keys.erase(it);    //remove prefix if it exceeds the size constrain
            }
            

        } else {
            it = keys.erase(it);    //remove prefix if it exceeds the delta constrain
        }
    }
    //add the new motifs to the current prefix list
    if (!new_motif.empty()) {
        for (vector<event> const &mt: new_motif) {
            keys.insert(mt);
        }
    }
    vector<event> E;
    E.push_back(e);
    imap[E].first += 1;
    imap[E].second.insert(u);
    imap[E].second.insert(v);
    keys.insert(E); // add the new event to the current prefix list
    Emap[e.first.first].insert(e.second);
    for (auto it=Emap.begin(); it!=Emap.end(); ) {
        if (e.first.first - it->first <=d_c) {
            ++it;
        } else {
            it = Emap.erase(it);
        }
    }
    return;
}

string encodeMotif(vector<event> instance){
    string motif;
    string temp;
    bool concurrent {false};
    timestamp t0 = instance[0].first.first - 1;
    map<vertex, string> code;
    string fraud_node_code = "";
    string fraud_edge_code = "";
    //fraud_info["vert"] = 0;
    //fraud_info["edge"] = 0;

    int i=0;
    for (auto it=instance.begin(); it!=instance.end(); ++it) {
        vertex u = it->second.first.first.first;
        bool u_label = it->second.first.first.second;
        vertex v = it->second.first.second.first;
        bool v_label = it->second.first.second.second;
        // cout<<u <<" "<<u_label<<" "<<v <<" "<<v_label<<endl;
        timestamp t = it->first.first;
        if (t==t0) {
            motif.append("{");
            concurrent = true;
        }
        motif.append(temp);
        temp.clear();
        if (t!=t0&&concurrent){
            motif.append("}");
            concurrent = false;
        }
        t0 = t;
        if (code.find(u)==code.end()) {
            code[u] = to_string(i);
            // if (u_label){
            //    fraud_info["vert"] += 1; 
            // }
            if (u_label){
                fraud_node_code.append("1"); 
            }
            else{
                fraud_node_code.append("0");
            }
            i++;
        }
        temp.append(code[u]);
        if (code.find(v)==code.end()) {
            code[v] = to_string(i);
            // if (v_label){
            //    fraud_info["vert"] += 1; 
            // }
            if (v_label){
                fraud_node_code.append("1"); 
            }
            else{
                fraud_node_code.append("0");
            }
            i++;
        }
        temp.append(code[v]);
        if (it->second.second){
            // fraud_info["edge"] += 1;
            fraud_edge_code.append("1");
        }
        else{
            fraud_edge_code.append("0");
        }
    }
    motif.append(temp);
    // motif.append(to_string(fraud_info["vert"]));
    // motif.append(to_string(fraud_info["edge"]));
    motif.append("|");
    motif.append(fraud_node_code);
    motif.append("|");
    motif.append(fraud_edge_code);
    if (concurrent) {
        motif.append("}");
    }
    return motif;
}

//string encodeMotif(vector<event> instance){
//    string motif;
//    map<vertex, string> code;
//    int i=0;
//    unordered_map<timestamp, int> concurrent_count;
//    for (auto it=instance.begin(); it!=instance.end(); ++it) {
//        vertex u = it->second.first;
//        vertex v = it->second.second;
//        timestamp t = it->first;
//        concurrent_count[t] += 1;
//        if (concurrent_count[t]==2) {
//            motif.append("a");
//        }
//        if (code.find(u)==code.end()) {
//            code[u] = to_string(i);
//            i++;
//        }
//        motif.append(code[u]);
//        if (code.find(v)==code.end()) {
//            code[v] = to_string(i);
//            i++;
//        }
//        motif.append(code[v]);
//        if (concurrent_count[t]>1) {
//            char c = sconvert(concurrent_count[t]);
//            motif.push_back(c);
//        }
//    }
//    return motif;
//}

char sconvert (int i) {
    string s("abcdefghijklmnopqrstuvwxyz");
    return s.at(i-1);
}

set<vertex> getNodes(vector<event> key){
    set<vertex> nodes;
    for (int i=0; i<key.size(); i++) {
        nodes.insert(key[i].second.first.first.first);
        nodes.insert(key[i].second.first.second.first);
    }
    return nodes;
}

bool checkInduced(eventmap Emap, event e, vector<timestamp> T){
    timestamp t = e.first.first;
    vertex u = e.second.first.first.first;
    vertex v = e.second.first.second.first;
    for (int i=0; i<T.size(); i++) {
        timestamp t0 = T[i];
        for (auto it = Emap[t0].begin(); it != Emap[t0].end();++it){
            if (it->first.first.first==u && it->first.second.first==v) {
                return false;
            }
        }
    }
    return true;
}

void countMotif (event e, set<key>& pre, map<string, int>& motif_count, int N_vtx, int N_event, int d_c, int d_w, eventmap& Emap){
    vertex u = e.second.first.first.first;
    vertex v = e.second.first.second.first;
    vector<vector<event>> new_motif;    //used to store the new motifs
    for (auto it = pre.begin(); it != pre.end();) {   //for each current prefix
        vector<event> key = *it;
        set<vertex> nodes = getNodes(key);
        if (e.first.first - key.front().first.first <= d_w && e.first.first - key.back().first.first <= d_c ) {  //check delta C and delta W
            if (key.size() < N_event){  //check n events
                if (nodes.find(u)!=nodes.end() || nodes.find(v)!=nodes.end()) {
                    nodes.insert(u);
                    nodes.insert(v);
                    if (key.back().first.first!=e.first.first) { //check synchronous events
                        if (nodes.size() <= N_vtx) {    //check the number of vertices
                            vector<event> motif = key;
                            vector<timestamp> T;
                            for (int i=0; i<motif.size(); i++) {
                                if (motif[i].second.first.first.first!=u||motif[i].second.first.second.first!=v) {
                                    T.push_back(motif[i].first.first);
                                }
                            }
                            if (checkInduced(Emap, e, T)) {
                                motif.push_back(e);
                                if (motif.size()==N_event && nodes.size()==N_vtx) {
                                    string code = encodeMotif(motif);
                                    motif_count[code] += 1;
                                } else if(motif.size()<N_event) {
                                    new_motif.push_back(motif);
                                }
                            }
                        }
                    }
                }
                ++it;
            } else {
                it = pre.erase(it);    //remove prefix if it exceeds N events constrain
            }
        } else {
            it = pre.erase(it);    //remove prefix if it exceeds the delta constrain
        }
    }
    //add the new motifs to the current prefix list
    if (!new_motif.empty()) {
        for (vector<event> const &mt: new_motif) {
            pre.insert(mt);
        }
    }
    vector<event> E;
    E.push_back(e);
    pre.insert(E); // add the new event to the current prefix list
    Emap[e.first.first].insert(e.second);
    for (auto it=Emap.begin(); it!=Emap.end(); ) {
        if (e.first.first - it->first <=d_c) {
            ++it;
        } else {
            it = Emap.erase(it);
        }
    }
    return;
}

void countSpecificmotif (event e, set<key>& pre, int& motif_count, string code_given, int N_vtx, int N_event, int d_c, int d_w){
    vector<vector<event>> new_motif;    //used to store the new motifs
    for (auto it = pre.begin(); it != pre.end();) {   //for each current prefix
        vector<event> key = *it;
        if (e.first.first - key.front().first.first <= d_w && e.first.first - key.back().first.first <= d_c) {  //check delta C and delta W
            vector<event> motif = key;
            motif.push_back(e);
            set<vertex> nodes = getNodes(motif);
            if (motif.size()==N_event && nodes.size()==N_vtx) {
                string code = encodeMotif(motif);
                int l = code.length();
                if (code==code_given.substr(0,l)) {
                    motif_count += 1;
                }
            } else if(motif.size()<N_event) {
                string code = encodeMotif(motif);
                int l = code.length();
                if (code==code_given.substr(0,l)) {
                    new_motif.push_back(motif);
                }
            }
            ++it;
        } else {
            it = pre.erase(it);    //remove prefix if it exceeds the delta constrain
        }
    }
    //add the new motifs to the current prefix list
    if (!new_motif.empty()) {
        for (vector<event> const &mt: new_motif) {
            pre.insert(mt);
        }
    }
    vector<event> E;
    E.push_back(e);
    pre.insert(E); // add the new event to the current prefix list
    return;
}
