Code.require_file("lib/candlex/mixfile.ex")

defmodule Candlex.MixProject do
  use Candlex.Mixfile

  defp extra_deps do
    [
      {:nx, "~> 0.6.0"}
    ]
  end
end
