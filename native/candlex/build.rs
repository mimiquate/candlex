fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    bindgen_cuda::Builder::default()
        .kernel_paths_glob("src/kernels/*.cu")
        .build_ptx()
        .unwrap()
        .write("src/kernels.rs")
        .unwrap();
}
