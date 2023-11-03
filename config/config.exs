import Config

config :candlex, use_cuda: System.get_env("CANDLEX_NIF_TARGET") == "cuda"
