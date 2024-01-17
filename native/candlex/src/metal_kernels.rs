use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineState, Device, Function, Library,
    MTLSize,
};
use std::ffi::c_void;

const CUSTOM_UNARY: &str = include_str!("metal_kernels/custom_unary.metal");
const CUSTOM_BINARY: &str = include_str!("metal_kernels/custom_binary.metal");

use candle_metal_kernels::MetalKernelError;

fn linear_split(pipeline: &ComputePipelineState, length: usize) -> (MTLSize, MTLSize) {
    let size = length as u64;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);

    let thread_group_count = MTLSize {
        width: (size + width - 1) / width,
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

    encoder.set_bytes(
        0,
        core::mem::size_of::<usize>() as u64,
        &length as *const usize as *const c_void,
    );
    encoder.set_buffer(1, Some(input_buffer), 0);
    encoder.set_buffer(2, Some(output_buffer), 0);

    encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    encoder.end_encoding();

    Ok(())
}

pub fn call_custom_unary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernel_name: custom_unary::strided::Kernel,
    shape: &[usize],
    strides: &[usize],
    input_buffer: &Buffer,
    input_offset: usize,
    output_buffer: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline =
        CustomKernels::new().load_pipeline(device, Source::CustomUnary, kernel_name.0)?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let num_dims: usize = shape.len();
    let length: usize = shape.iter().product();

    encoder.set_bytes(
        0,
        core::mem::size_of::<usize>() as u64,
        &length as *const usize as *const c_void,
    );
    encoder.set_bytes(
        1,
        core::mem::size_of::<usize>() as u64,
        &num_dims as *const usize as *const c_void,
    );

    encoder.set_bytes(
        2,
        core::mem::size_of_val(shape) as u64,
        shape.as_ptr() as *const c_void,
    );

    encoder.set_bytes(
        3,
        core::mem::size_of_val(strides) as u64,
        strides.as_ptr() as *const c_void,
    );

    encoder.set_buffer(4, Some(input_buffer), input_offset as u64);
    encoder.set_buffer(5, Some(output_buffer), output_offset as u64);

    encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let width: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);
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

    encoder.set_bytes(
        0,
        core::mem::size_of::<usize>() as u64,
        &length as *const usize as *const c_void,
    );
    encoder.set_buffer(1, Some(left_buffer), 0);
    encoder.set_buffer(2, Some(right_buffer), 0);
    encoder.set_buffer(3, Some(output_buffer), 0);

    encoder.use_resource(left_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    encoder.end_encoding();

    Ok(())
}

pub fn call_custom_binary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernel_name: custom_binary::strided::Kernel,
    shape: &[usize],
    left_buffer: &Buffer,
    left_strides: &[usize],
    left_offset: usize,
    right_buffer: &Buffer,
    right_strides: &[usize],
    right_offset: usize,
    output_buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = CustomKernels::new().load_pipeline(device, Source::CustomBinary, kernel_name.0)?;

    let num_dims: usize = shape.len();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();

    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            left_strides,
            right_strides,
            (left_buffer, left_offset),
            (right_buffer, right_offset),
            output_buffer
        )
    );

    encoder.use_resource(left_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(right_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);

    let width: usize = shape.iter().product();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, width);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    encoder.end_encoding();

    Ok(())
}
