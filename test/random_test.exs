defmodule Candlex.RandomTest do
  use Nx.Case, async: true

  test "key/1" do
    Nx.Random.key(42)
    |> assert_equal(Nx.tensor([0, 42]))
  end

  test "uniform/1" do
    {normal, new_key} =
      Nx.Random.key(42)
      |> Nx.Random.uniform()

    normal
    |> assert_close(Nx.tensor(0.9145736694335938))

    new_key
    |> assert_equal(Nx.tensor([2_465_931_498, 3_679_230_171]))
  end

  test "normal/1" do
    {normal, new_key} =
      Nx.Random.key(42)
      |> Nx.Random.normal()

    normal
    |> assert_close(Nx.tensor(1.3694695234298706))

    new_key
    |> assert_equal(Nx.tensor([2_465_931_498, 3_679_230_171]))
  end
end
