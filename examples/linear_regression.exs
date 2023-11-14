defmodule LinearRegression do
  import Nx.Defn

  defn init_random_params do
    {m, new_key} =
      Nx.Random.key(42)
      |> Nx.Random.normal(0.0, 0.1, shape: {1, 1})

    {b, _new_key} =
      new_key
      |> Nx.Random.normal(0.0, 0.1, shape: {1})

    {m, b}
  end

  defn predict({m, b}, input) do
    Nx.dot(input, m) + b
  end

  # MSE Loss
  defn loss({m, b}, input, target) do
    target - predict({m, b}, input)
    |> Nx.pow(2)
    |> Nx.mean()
  end

  defn update({m, b} = params, input, target, step) do
    {grad_m, grad_b} =
      params
      |> grad(&loss(&1, input, target))

    {
      m - grad_m * step,
      b - grad_b * step
    }
  end

  def train(params, epochs, lin_fn) do
    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, lin_fn)) end)

    for _ <- 1..epochs, reduce: params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {input, target} = Enum.unzip(batch)
            x = Nx.reshape(Nx.tensor(input), {32, 1})
            y = Nx.reshape(Nx.tensor(target), {32, 1})
            update(cur_params, x, y, 0.001)
          end
        )
    end
  end
end

Nx.default_backend(Candlex.Backend)

params = LinearRegression.init_random_params()
m = :rand.normal(0.0, 10.0)
b = :rand.normal(0.0, 5.0)
IO.puts("Target m: #{m} Target b: #{b}\n")

lin_fn = fn x -> m * x + b end
epochs = 100

# These will be very close to the above coefficients
{time, {trained_m, trained_b}} = :timer.tc(LinearRegression, :train, [params, epochs, lin_fn])

trained_m =
  trained_m
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

trained_b =
  trained_b
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

IO.puts("Trained in #{time / 1_000_000} sec.")
IO.puts("Trained m: #{trained_m} Trained b: #{trained_b}\n")
IO.puts("Accuracy m: #{m - trained_m} Accuracy b: #{b - trained_b}")
