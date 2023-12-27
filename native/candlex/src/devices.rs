#[rustler::nif(schedule = "DirtyCpu")]
pub fn is_cuda_available() -> bool {
    candle_core::utils::cuda_is_available()
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn is_metal_available() -> bool {
    candle_core::utils::metal_is_available()
}
