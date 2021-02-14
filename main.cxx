#include "MyDataset.hh"
#include <iostream>
#include <memory>
#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(2, 4));
        fc2 = register_module("fc2", torch::nn::Linear(4, 4));
        fc3 = register_module("fc3", torch::nn::Linear(4, 2));
    }
    torch::nn::Linear fc1 = {nullptr};
    torch::nn::Linear fc2 = {nullptr};
    torch::nn::Linear fc3 = {nullptr};
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::softmax(fc3->forward(x), 1);
        return x;
    }
};

int main() {
    std::shared_ptr<DataStorage> data_storage = std::make_shared<DataStorage>();
    data_storage->MakeData();
    auto data_set = CustomDataSet(data_storage.get()).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set),
                                                                                               /*batch_size = */ 4);
    int n_epochs = 4000;
    auto net = std::make_shared<Net>();

    // torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-3));
    torch::optim::Adam optimizer(net->parameters());

    for (int epoch = 1; epoch <= n_epochs; ++epoch) {
        for (auto &batch : *data_loader) {
            auto data = batch.data;
            auto target = batch.target;
            optimizer.zero_grad();
            auto output = net->forward(data);
            auto loss = torch::binary_cross_entropy(output, target);
            loss.backward();
            optimizer.step();
            std::cout << "epoch = " << epoch << " Loss = " << loss.item<float>() << std::endl;
        }
    }
    auto input_tensor = torch::from_blob(data_storage->inputs_1d.data(), {4, 2});
    std::cout << net->forward(input_tensor) << std::endl;
    return 0;
}
