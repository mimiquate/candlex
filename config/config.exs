import Config

config :candlex, use_cuda: (System.get_env("NATIVE_TARGET") == "cuda")
