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

  describe "while/3" do
    defmodule Mod do
      import Nx.Defn

      defn upto10(x) do
        while x, Nx.less(x, 10) do
          x + 1
        end
      end

      defn factorial_tuple(x) do
        factorial = Nx.tensor(1, type: Nx.type(x))

        {factorial, _} =
          while {factorial, x}, Nx.greater(x, 1) do
            {factorial * x, x - 1}
          end

        factorial
      end

      defn factorial_map(x) do
        factorial = Nx.tensor(1, type: Nx.type(x))

        %{factorial: factorial} =
          while map = %{factorial: factorial, x: x}, Nx.greater(map.x, 1) do
            %{map | factorial: map.factorial * map.x, x: map.x - 1}
          end

        factorial
      end

      defn factorial_map_input(map) do
        %{factorial: factorial} =
          while map, Nx.greater(map.x, 1) do
            %{map | factorial: map.factorial * map.x, x: map.x - 1}
          end

        factorial
      end

      defn tensor_generator_sum() do
        while x = 0, r <- Nx.tensor([0, 1, 2]) do
          x + r
        end
      end
    end

    test "simple" do
      Mod.upto10(0)
      |> assert_equal(Nx.tensor(10))

      Mod.upto10(5)
      |> assert_equal(Nx.tensor(10))
    end

    test "factorial tuple" do
      Mod.factorial_tuple(5)
      |> assert_equal(Nx.tensor(120))

      Mod.factorial_tuple(10.0)
      |> assert_equal(Nx.tensor(3_628_800.0))
    end

    test "factorial map" do
      Mod.factorial_map(5)
      |> assert_equal(Nx.tensor(120))

      Mod.factorial_map(10.0)
      |> assert_equal(Nx.tensor(3_628_800.0))
    end

    test "factorial map input" do
      Mod.factorial_map_input(%{factorial: 1, x: 5})
      |> assert_equal(Nx.tensor(120))

      Mod.factorial_map_input(%{factorial: 1.0, x: 10.0})
      |> assert_equal(Nx.tensor(3_628_800.0))
    end

    test "tensor generator sum" do
      Mod.tensor_generator_sum()
      |> assert_equal(Nx.tensor(3))
    end
  end
end
