defmodule Candlex.DefnTest do
  use Nx.Case, async: true

  test "grad/1" do
    grad_fun = Nx.Defn.grad(fn x -> Nx.sin(x) end)

    grad_fun.(Nx.tensor(0.0))
    |> assert_equal(Nx.tensor(1.0))
  end

  test "grad/2" do
    defmodule TG do
      import Nx.Defn

      defn tanh_grad(t) do
        grad(t, &Nx.tanh/1)
      end
    end

    Nx.tensor(1.0)
    |> TG.tanh_grad()
    |> assert_close(Nx.tensor(0.41997432708740234))
  end
end
