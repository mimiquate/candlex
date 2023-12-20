Code.compile_file("lib/candlex/mixfile.ex")

defmodule Candlex.MixProject do
  use Candlex.Mixfile

  defp extra_deps do
    [
      {:nx, "~> 0.5.0"},
      {:bumblebee, "~> 0.3.0"}
    ]
  end
end
