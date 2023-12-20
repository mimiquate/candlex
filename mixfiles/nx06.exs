Code.eval_file("mixfile.ex")

defmodule Candlex.MixProject do
  use Candlex.Mixfile

  defp extra_deps do
    [
      {:nx, "~> 0.6.0"}
    ]
  end
end
