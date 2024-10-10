mod atoms {
    rustler::atoms! {
        cpu,
        cuda,
        metal
    }
}

mod devices;
mod error;
#[cfg(feature = "cuda")]
mod kernels;
mod ops;
mod tensors;

use rustler::{Env, Term};
use tensors::TensorRef;

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(TensorRef, env);
    true
}

rustler::init! {
    "Elixir.Candlex.Native",
    load = load
}
