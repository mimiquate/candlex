Code.compile_file("project.exs")

defmodule Candlex.MixProject do
  use Mix.Project

  def project do
    Candlex.Project.project()
    |> Keyword.put(:deps, new_deps())
    |> Keyword.put(:lockfile, lockfile())
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    Candlex.Project.application()
  end

  defp new_deps do
    extra_deps()
    |> Enum.reduce(
      Candlex.Project.deps(),
      fn extra_dep, deps ->
        deps
        |> List.keyreplace(elem(extra_dep,0), 0, extra_dep)
      end
    )
  end

  defp extra_deps do
    [
      {:nx, "~> 0.6.0"}
    ]
  end

  defp lockfile do
    Path.join(__DIR__, "#{Path.basename(__ENV__.file, ".exs")}.lock")
  end
end
