#ifndef _MyDataset_
#define _MyDataset_

#include "DataStorage.hh"
#include <torch/torch.h>
#include <vector>

class CustomDataSet : public torch::data::Dataset<CustomDataSet> {
  private:
    torch::Tensor inputs, labels;

  public:
    CustomDataSet(DataStorage *data_strage) {
        int n_data = data_strage->n_data;
        int n_input = data_strage->n_input;
        int n_output = data_strage->n_output;
        inputs = torch::from_blob(data_strage->inputs_1d.data(), {n_data, n_input});
        labels = torch::from_blob(data_strage->labels_1d.data(), {n_data, n_output});
    }
    torch::data::Example<> get(std::size_t index) override { return {inputs[index], labels[index]}; }
    torch::optional<std::size_t> size() const override { return labels.size(0); }
};

#endif
