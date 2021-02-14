#ifndef _DataStorage_
#define _DataStorage_

#include <vector>

struct DataStorage {
    std::vector<float> inputs_1d;
    std::vector<float> labels_1d;
    const int n_input = 2;
    const int n_output = 2;
    int n_data = 4;
    void MakeData();
};

void DataStorage::MakeData() {
    inputs_1d = {0, 0, 0, 1, 1, 0, 1, 1};
    labels_1d = {0, 1, 1, 0, 1, 0, 0, 1};
}

#endif
