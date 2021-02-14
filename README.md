# Libtorch_example_2
Some examples of Libtorch. (Part 2)
Libtorch_example_1の続き。やってることはほとんど同じだけど、CustomDataSetを作ってやっているという点が異なる。
これをすることによって、batchごとの学習ができるようになったりする。多分他にも色々とできるようになると思われる。

CustomDataSet
-
```c++
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
```
本来はこのCustomDataSetを自分で定義してデータ等にアクセスできるようにするのが一般的なはず。CustomDataSetは`torch::data::Dataset<CustomDataSet>` から継承して、メンバ関数`torch::data::Example<> get()` と `torch::optional<std::size_t> size()` をオーバーライドしないといけない。`get()`では、`index` 番目のデータを input, output 共に `torch::Tensor` 型で返す必要がある。`size()` は(多分)データ数を返す必要あり。

データの取り出し方
-
main 関数の中にある
```c++
    std::shared_ptr<DataStorage> data_storage = std::make_shared<DataStorage>();
    data_storage->MakeData();
    auto data_set = CustomDataSet(data_storage.get()).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), /*batch_size = */ 4);
```
の部分で、データをロードする。`auto data_set = CustomDataSet(data_storage.get()).map(torch::data::transforms::Stack<>());`
の部分はコンストラクタ以外は写経でOK。その次の、`auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(data_set), /*batch_size = */ 4);`
の部分でデータをロードしていると思われる。第２引数で batch size を決める。(2だったらデータを2個ずつに分けてそれぞれで学習をする。確か。)
そのあとの、for 文で
`for (auto & batch : *data_loader){}` とすることで、各 batch のデータを見に行くことができる。
実際に、`auto data = batch.data` と `auto target = batch.target` でそれぞれ入力データとラベルを取ってきている。

出力
-

```
 0.0010  0.9990
 0.9819  0.0181
 0.9998  0.0002
 0.0010  0.9990
[ CPUFloatType{4,2} ]
```