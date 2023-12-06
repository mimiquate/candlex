Nx.default_backend(Candlex.Backend)

{:ok, model_info} = Bumblebee.load_model({:hf, "microsoft/resnet-50"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, "microsoft/resnet-50"})

%{predictions: [%{label: "tiger cat", score: score}]} =
  Bumblebee.Vision.image_classification(model_info, featurizer, top_k: 1)
  |> Nx.Serving.run(StbImage.read_file!(Path.join(__DIR__, "images/tiger-cat.jpeg")))
  |> IO.inspect()

true = abs(score - 0.8908) < 0.0001
