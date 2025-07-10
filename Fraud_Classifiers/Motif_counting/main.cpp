//
//  main.cpp
//  tmc
//
//

#include <iostream>
#include "tmc.hpp"
// #include <cmath>

int main(int argc, char * argv[]) {
	const auto t1 = chrono::steady_clock::now();

//    string method (argv[1]);
//    if (!(method == "v1" || method == "v2" || method == "v3")) {
//        printf ("Invalid algorithm, options are v1:count all motifs for given size, v2:query a specific type of motif\n");
//        exit(1);
//    }

	string tmp (argv[1]);
	string gname = tmp.substr (tmp.find_last_of("/") + 1);
	int d_c=stoi(argv[2]);      //delta C
	int d_w=stoi(argv[3]);      //delta W
    int N_vtx=stoi(argv[4]);    //number of vertices in the motif
    int N_event=stoi(argv[5]);  //number of events in the motif
    int bin=stoi(argv[6]);      //resolution
    string consecutive (argv[7]);   // consecutive event restriction? (YES/NO)
    string dgc (argv[8]);   // constrained dynamic graphlet counting? (YES/NO)
    string amount_constraint (argv[9]);
    int a_c=stoi(argv[10]);      //amount C
    int a_w=stoi(argv[11]);      //amount W

	string out_file;
    out_file = "../Output/" + gname + "_" + argv[2] + "_" + argv[3] + "_" + argv[4] + "_" + argv[5]+".txt";
	FILE* fp = fopen (out_file.c_str(), "w");

	cout << "delta C: " << d_c << endl;
	cout << "delta W: " << d_w << endl;


	//Read file and create a sorted list of temporal events
	vector<event> events;
	createEvents(tmp, events, bin);
	const auto t2 = chrono::steady_clock::now();
	print_time (fp, "Read data time: ", t2 - t1);
	cout << "# of events: " << events.size() << endl;

    //Enumerate all instances that satisfy the given constrains(delta C, delta W, and motif size)
    instancemap instances;
    set<vector<event>> keys;
    eventmap Emap;
    for (size_t i=0; i<events.size(); i++) {
        countInstance(events[i], instances, keys, N_vtx, N_event, d_c, d_w, consecutive, dgc, amount_constraint, a_c, a_w,  Emap);
    }

    const auto t3 = chrono::steady_clock::now();
    print_time (fp, "Count instances time: ", t3 - t2);
    cout << "# of instances: " << instances.size() << endl;

    //Classify instances to corresponding type of motif
    map<string, int> motif_count;
    // string out_file1 = out_file + "_time.txt";
    string out_file1 = "Motif_amount.txt";
    FILE* fp1 = fopen(out_file1.c_str(), "a");
    for (auto it=instances.begin(); it!=instances.end(); ++it) {
        if (it->first.size()==N_event && it->second.second.size()==N_vtx) {
            string motif = encodeMotif(it->first);
            vector<event> motif_events = it->first;
            if (motif_events.size()<2) continue;
            vector<double> amount;
            fprintf (fp1, "%s,", motif.c_str());
            for (int i=0; i<motif_events.size(); i++) {
                amount.push_back(motif_events[i].first.second);
                fprintf (fp1, "%d,", motif_events[i].first.second);
            }
            fprintf (fp1, "\n");
            double mean = accumulate(amount.begin(), amount.end(), 0.0) / amount.size();
            double var = 0;
            for(int i = 0; i < amount.size(); i++) {
                var += pow(amount[i] - mean, 2);
            }
            double std = sqrt(var / amount.size());
            if (std / mean < 0.5) {
                motif_count["S"+motif] += it->second.first;
            } else if (std / mean < 1) {
                motif_count["M"+motif] += it->second.first;
            } else {
                motif_count["L"+motif] += it->second.first;
            }
            // Only consider delta_C
            // if (N_event == 3) {
            //     int second_event_percentile = 100 * ((double) (motif_events[1].first.first - motif_events[0].first.first) / (motif_events[2].first.first - motif_events[0].first.first));
            //     fprintf (fp1, "%s %d %d %d perc dc: %d\n", motif.c_str(), 0, second_event_percentile, 100, d_c);
            //     fprintf (fp1, "%s %d span dc: %d\n", motif.c_str(), (motif_events[2].first.first - motif_events[0].first.first), d_c);
            // }
            // if (N_event == 4) {
            //     int second_event_percentile = 100 * ((double) (motif_events[1].first.first - motif_events[0].first.first) / (motif_events[3].first.first - motif_events[0].first.first));
            //     int third_event_percentile = 100 * ((double) (motif_events[2].first.first - motif_events[0].first.first) / (motif_events[3].first.first - motif_events[0].first.first));
            //     fprintf (fp1, "%s %d %d %d %d perc dc: %d\n", motif.c_str(), 0, second_event_percentile, third_event_percentile, 100, d_c);
            //     fprintf (fp1, "%s %d span dc: %d\n", motif.c_str(), (motif_events[3].first.first - motif_events[0].first.first), d_c);
            // }
            motif_count[motif] += it->second.first;
        }
    }

    const auto t4 = chrono::steady_clock::now();
    print_time (fp, "Count motifs time: ", t4 - t3);
    print_time (fp, "End-to-end Time: ", t4 - t1);

    for (auto it=motif_count.begin(); it!=motif_count.end(); ++it) {
        fprintf(fp, "%s \t %d \n", (*it).first.c_str(), (*it).second);
    }
    fclose (fp);


//        if (method=="v2") {
//            N_vtx=stoi(argv[5]);    //number of vertices in the motif
//            N_event=stoi(argv[6]);  //number of events in the motif
//            //Directly count motif from events stream
//            map<string, int> motif_count;
//            set<key> pre;
//            eventmap Emap;
//            for (size_t i=0; i<events.size(); i++) {
//                countMotif(events[i], pre, motif_count, N_vtx, N_event, d_c, d_w, Emap);
////                Emap[events[i].first].insert(events[i].second);
//            }
//            // removeIsomorphic(motif_count);
//
//            const auto t4 = chrono::steady_clock::now();
//            print_time (fp, "Count motifs time: ", t4 - t2);
//            print_time (fp, "End-to-end Time: ", t4 - t1);
//
//            for (auto it=motif_count.begin(); it!=motif_count.end(); ++it) {
//                fprintf(fp, "%s \t %d \n", (*it).first.c_str(), (*it).second);
//            }
//            fclose (fp);
//        }
//
//    if (method=="v3") {
//        string motif (argv[5]);
//        string edges;
//        for (int i=0; i<motif.size(); i++) {
//            if (motif[i]>='0'&&motif[i]<='9') {
//                edges.push_back(motif[i]);
//            }
//        }
//        int l = edges.length();
//        if (l%2!=0) {
//            printf ("Invalid input, input example: 011202\n");
//            exit(1);
//        }
//        N_event = l/2;
//        N_vtx = 0;
//        for (int i=0; i<edges.length(); i++) {
//            int a = edges[i] - '0';
//            N_vtx = max(a, N_vtx);
//        }
//        N_vtx++;
//        int motif_count;
//        set<key> pre;
//        for (size_t i=0; i<events.size(); i++) {
//            countSpecificmotif (events[i], pre, motif_count, motif, N_vtx, N_event, d_c, d_w);
//        }
//        const auto t3 = chrono::steady_clock::now();
//        print_time (fp, "Count motif time: ", t3 - t2);
//        print_time (fp, "End-to-end Time: ", t3 - t1);
//        cout << "result: " << motif << "\t" << motif_count << endl;
//        fprintf(fp, "%s \t %d \n", motif.c_str(), motif_count);
//    }

	return 0;
}
