# Candlex

[![ci](https://github.com/mimiquate/candlex/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mimiquate/candlex/actions?query=branch%3Amain)
[![Hex.pm](https://img.shields.io/hexpm/v/candlex.svg)](https://hex.pm/packages/candlex)
[![Docs](https://img.shields.io/badge/docs-gray.svg)](https://hexdocs.pm/candlex)

An `Nx` [backend](https://hexdocs.pm/nx/Nx.html#module-backends) for [candle](https://huggingface.github.io/candle) machine learning minimalist framework

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `candlex` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:candlex, "~> 0.1.1"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/candlex>.

## Releasing

To publish a new version of this package:

1. Update `@version` in `mix.exs`.
1. `git tag -s <tag-version>` to create new signed tag.
1. `git push origin <tag-version>` to push the tag.
1. Wait for the `binaries.yml` GitHub workflow to build all the NIF binaries.
1. `mix rustler_precompiled.download Candlex.Native --all` to generate binaries checksums locally.
1. `mix hex.build --unpack` to check the package includes the correct files.
1. Publish the release from draft in GitHub.
1. `mix hex.publish` to publish package to Hex.pm.

## License

Copyright 2023 Mimiquate

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
