Code.require_file("lib/candlex/mixfile.ex")

defmodule Candlex.MixProject do
  use Candlex.Mixfile

  defp extra_deps do
    [
      {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
    ]
  end
end
