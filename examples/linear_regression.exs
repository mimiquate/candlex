defmodule LinearRegression do
  import Nx.Defn

  @epochs 100
  @step 0.001 # Sometimes also called learning rate

  defn initial_random_params do
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

  defn mse_loss(params, input, target) do
    target - predict(params, input)
    |> Nx.pow(2)
    |> Nx.mean()
  end

  defn update({m, b} = params, input, target) do
    {grad_m, grad_b} =
      params
      |> grad(&mse_loss(&1, input, target))

    {
      m - grad_m * @step,
      b - grad_b * @step
    }
  end

  def train(params, linear_fn) do
    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, linear_fn)) end)

    for _ <- 1..@epochs, reduce: params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, current_params ->
            {input, target} = Enum.unzip(batch)

            update(
              current_params,
              Nx.reshape(Nx.tensor(input), {32, 1}),
              Nx.reshape(Nx.tensor(target), {32, 1})
            )
          end
        )
    end
  end
end

Nx.default_backend(Candlex.Backend)

initial_params = LinearRegression.initial_random_params()
m = :rand.normal(0.0, 10.0)
b = :rand.normal(0.0, 5.0)
IO.puts("Target m: #{m} Target b: #{b}\n")

linear_fn = fn x -> m * x + b end

# These will be very close to the above coefficients
{time, {trained_m, trained_b}} = :timer.tc(LinearRegression, :train, [initial_params, linear_fn])

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
