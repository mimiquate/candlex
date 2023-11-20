defmodule LinearRegression do
  import Nx.Defn

  @epochs 100
  @gradient_step_size 0.001 # Sometimes also called "learning rate"

  def fit(linear_fn) do
    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, linear_fn)) end)

    for _ <- 1..@epochs, reduce: initial_random_params() do
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

  defnp initial_random_params do
    {m, new_key} =
      Nx.Random.key(42)
      |> Nx.Random.normal(0.0, 0.1, shape: {1, 1})

    {b, _new_key} =
      new_key
      |> Nx.Random.normal(0.0, 0.1, shape: {1})

    {m, b}
  end

  defnp evaluate({m, b}, input) do
    Nx.dot(input, m) + b
  end

  defnp mean_squared_error(params, input, target) do
    target - evaluate(params, input)
    |> Nx.pow(2)
    |> Nx.mean()
  end

  defnp update({m, b} = params, input, target) do
    {grad_m, grad_b} =
      params
      |> grad(&mean_squared_error(&1, input, target))

    {
      m - grad_m * @gradient_step_size,
      b - grad_b * @gradient_step_size
    }
  end
end

Nx.default_backend(Candlex.Backend)

m = :rand.normal(0.0, 10.0)
b = :rand.normal(0.0, 5.0)
IO.puts("Target m: #{m} Target b: #{b}\n")

# These should be very close to the above coefficients
{time, {fitted_m, fitted_b}} = :timer.tc(LinearRegression, :fit, [fn x -> m * x + b end])

fitted_m =
  fitted_m
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

fitted_b =
  fitted_b
  |> Nx.squeeze()
  |> Nx.backend_transfer()
  |> Nx.to_number()

IO.puts("Fitted in #{time / 1_000_000} sec.")
IO.puts("Fitted m: #{fitted_m} Fitted b: #{fitted_b}\n")
IO.puts("Accuracy m: #{m - fitted_m} Accuracy b: #{b - fitted_b}")
