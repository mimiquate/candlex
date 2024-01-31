defmodule Candlex.MixProject do
  use Mix.Project

  @description "An Nx backend for candle machine learning minimalist framework"
  @source_url "https://github.com/mimiquate/candlex"
  @version "0.1.8"
  @blend_dir "blend"

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
      package: package(),
      lockfile: lockfile(),
      build_path: build_path(),
      deps_path: deps_path()
    ]
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
      {:blend, "~> 0.1.0", only: :dev},
      {:bumblebee, "~> 0.3", only: :dev, runtime: false},
      {:stb_image, "~> 0.6", only: :dev, runtime: false},
      {:ex_doc, "~> 0.31.0", only: :dev, runtime: false}
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

  defp lockfile do
    if blend() do
      "#{@blend_dir}/#{blend()}.mix.lock"
    else
      "mix.lock"
    end
  end

  defp build_path do
    if blend() do
      Path.join([__DIR__, @blend_dir, "_build", blend()])
    else
      "_build"
    end
  end

  defp deps_path do
    if blend() do
      Path.join([__DIR__, @blend_dir, "deps", blend()])
    else
      "deps"
    end
  end

  defp blend do
    case System.get_env("BLEND") do
      "" -> nil
      blend -> blend
    end
  end
end
