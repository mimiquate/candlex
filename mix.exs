if File.exists?("blend/premix.exs") do
  Code.compile_file("blend/premix.exs")
end

defmodule Candlex.MixProject do
  use Mix.Project

  @description "An Nx backend for candle machine learning minimalist framework"
  @source_url "https://github.com/mimiquate/candlex"
  @version "0.1.10"

  def project do
    [
      app: :candlex,
      description: @description,
      version: @version,
      elixir: "~> 1.14",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
    |> Keyword.merge(maybe_lockfile_option())
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.6"},
      {:rustler_precompiled, "~> 0.7.0"},

      # Optional
      {:rustler, "~> 0.29", optional: true},

      # Dev
      {:blend, "~> 0.4.0", only: :dev},
      {:bumblebee, "~> 0.4", only: :dev, runtime: false},
      {:stb_image, "~> 0.6", only: :dev, runtime: false},
      {:ex_doc, "~> 0.36.0", only: :dev, runtime: false}
    ]
  end

  defp docs do
    [
      main: "Candlex",
      source_url: @source_url,
      source_ref: "v#{@version}"
    ]
  end

  defp package do
    [
      files: [
        "lib",
        "native",
        "priv",
        ".formatter.exs",
        "mix.exs",
        "CHANGELOG.md",
        "README.md",
        "LICENSE",
        "checksum-*.exs"
      ],
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url
      }
    ]
  end

  defp maybe_lockfile_option do
    case System.get_env("MIX_LOCKFILE") do
      nil -> []
      "" -> []
      lockfile -> [lockfile: lockfile]
    end
  end
end
