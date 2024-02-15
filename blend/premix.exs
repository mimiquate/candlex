maybe_put_env = fn varname, value ->
  System.put_env(varname, System.get_env(varname, value))
end

blend = System.get_env("BLEND")

if blend && String.length(blend) > 0 do
  maybe_put_env.("MIX_LOCKFILE", "blend/#{blend}.mix.lock")
  maybe_put_env.("MIX_DEPS_PATH", "blend/deps/#{blend}")
  maybe_put_env.("MIX_BUILD_ROOT", "blend/_build/#{blend}")
end
