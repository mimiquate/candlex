use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, Function, FunctionConstantValues, Library, MTLDataType, MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

const CUSTOM_UNARY: &str = include_str!("metal_kernels/custom_unary.metal");
const CUSTOM_BINARY: &str = include_str!("metal_kernels/custom_binary.metal");

use candle_metal_kernels::MetalKernelError;

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
fn linear_split(pipeline: &ComputePipelineState, length: usize) -> (MTLSize, MTLSize) {
    let size = length as u64;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = (size + width - 1) / width;
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    (thread_group_count, thread_group_size)
}

fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoderRef, position: u64, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
                encoder.set_bytes(
                    position,
                    core::mem::size_of::<$type>() as u64,
                    &data as *const $type as *const c_void,
                );
            }
        }
    };
}
primitive!(usize);
// primitive!(u32);
primitive!(f32);

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_bytes(
            position,
            core::mem::size_of_val(data) as u64,
            data.as_ptr() as *const c_void,
        );
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}
impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}
impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}

macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    CustomUnary,
    CustomBinary,
}

macro_rules! ops {
    ($($name:ident),+) => {
        pub mod contiguous {
            pub struct Kernel(pub &'static str);

            $(
                pub mod $name {
                    use super::Kernel;

                    pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32"));
                    pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
                    pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
                }
            )+
        }

        pub mod strided {
            pub struct Kernel(pub &'static str);

            $(
                pub mod $name {
                    use super::Kernel;
                    pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_f32_strided"));
                }
            )+
        }
    }
}

pub mod custom_unary {
    ops!(
        acos, acosh, asin, asinh, atan, atanh, bit_not, cbrt, cosh, erfc, erf_inv, expm1, is_inf,
        is_nan, ln_1p, sigmoid, sign, sinh, tan
    );
}

pub mod custom_binary {
    ops!(
        atan2,
        bit_and,
        bit_or,
        bit_xor,
        logical_and,
        logical_or,
        logical_xor,
        pow,
        remainder,
        shl,
        shr
    );
}

#[derive(Debug)]
pub struct CustomKernels {}

impl CustomKernels {
    pub fn new() -> Self {
        Self {}
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::CustomUnary => CUSTOM_UNARY,
            Source::CustomBinary => CUSTOM_BINARY,
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let lib = device
            .new_library_with_source(self.get_library_source(source), &CompileOptions::new())
            .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?;

        Ok(lib)
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, None)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;

        Ok(func)
    }

    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let func = self.load_function(device, source, name)?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;

        Ok(pipeline)
    }
}

pub fn call_custom_unary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernel_name: custom_unary::contiguous::Kernel,
    length: usize,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernels = CustomKernels::new();
    let pipeline = kernels.load_pipeline(device, Source::CustomUnary, kernel_name.0)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, input_buffer, output_buffer));

    encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    encoder.end_encoding();

    Ok(())
}

pub fn call_custom_binary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernel_name: custom_binary::contiguous::Kernel,
    length: usize,
    left_buffer: &Buffer,
    right_buffer: &Buffer,
    output_buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernels = CustomKernels::new();
    let pipeline = kernels.load_pipeline(device, Source::CustomBinary, kernel_name.0)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, left_buffer, right_buffer, output_buffer));

    encoder.use_resource(left_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    encoder.end_encoding();

    Ok(())
}
