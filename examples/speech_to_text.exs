Nx.default_backend(Candlex.Backend)

hf_repo = "openai/whisper-tiny"

{:ok, model_info} = Bumblebee.load_model({:hf, hf_repo})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, hf_repo})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, hf_repo})
{:ok, generation_config} = Bumblebee.load_generation_config({:hf, hf_repo})

%{chunks: [%{text: " Tower of strength."}]} =
  Bumblebee.Audio.speech_to_text_whisper(
    model_info,
    featurizer,
    tokenizer,
    generation_config
  )
  |> Nx.Serving.run(
    Path.join(__DIR__, "tower_of_strength_pcm_f32le_16000.bin")
    |> File.read!()
    |> Nx.from_binary(:f32)
  )
  |> IO.inspect()
