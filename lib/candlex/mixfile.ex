Code.require_file("project.exs")

defmodule Candlex.Mixfile do
  defmacro __using__(_opts) do
    quote do
      use Mix.Project

      def project do
        Candlex.Project.project()
        |> Keyword.put(:build_path, build_path())
        |> Keyword.put(:deps, new_deps())
        |> Keyword.put(:deps_path, deps_path())
        |> Keyword.put(:lockfile, lockfile())
      end

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

      defp build_path do
        Path.join([__DIR__, "_build", file_name()])
      end

      defp deps_path do
        Path.join([__DIR__, "deps", file_name()])
      end

      defp lockfile do
        Path.join(__DIR__, "#{file_name()}.lock")
      end

      defp file_name do
        Path.basename(__ENV__.file, ".exs")
      end
    end
  end
end
