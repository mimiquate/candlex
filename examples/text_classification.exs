Nx.default_backend(Candlex.Backend)

{:ok, model_info} = Bumblebee.load_model({:hf, "finiteautomata/bertweet-base-emotion-analysis"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "vinai/bertweet-base"})

%{predictions: [%{label: "joy", score: score}]} =
  Bumblebee.Text.text_classification(model_info, tokenizer, top_k: 1)
  |> Nx.Serving.run("I had a wonderful day")
  |> IO.inspect()

true = abs(score - 0.978) < 0.0001
