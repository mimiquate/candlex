Code.compile_file("project.exs")

defmodule Candlex.MixProject do
  use Mix.Project

  def project do
    Candlex.Project.project()
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    Candlex.Project.application()
  end
end
