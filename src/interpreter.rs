//! API of TensorFlow Lite [`Interpreter`] that performs inference.
use std::ffi::{c_void, CString};
use std::os::raw::c_int;

use crate::bindings::*;
use crate::model::Model;
use crate::tensor;
use crate::tensor::Tensor;
use crate::{Error, ErrorKind, Result};
use std::fmt::{Debug, Formatter};

/// Options for configuring the [`Interpreter`].
#[derive(Debug, Eq, PartialEq, Clone, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum Options {
    Default,
    #[cfg(feature = "xnnpack")]
    Xnnpack(i32),
    #[cfg(feature = "external_delegate")]
    External(String),
}

pub struct Interpreter<'a> {
    /// The configuration options for the [`Interpreter`].
    options: Options,

    /// The underlying [`TfLiteInterpreter`] C pointer.
    interpreter_ptr: *mut TfLiteInterpreter,

    /// The optional underlying [`TfLiteDelegate`] C pointer.
    delegate_ptr: Option<*mut TfLiteDelegate>,

    /// The underlying `Model` to limit lifetime of the interpreter.
    /// See this issue for details:
    /// <https://github.com/tensorflow/tensorflow/issues/53628>
    model: &'a Model<'a>,
}

impl Debug for Interpreter<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interpreter")
            .field("options", &self.options)
            .finish()
    }
}
unsafe impl Send for Interpreter<'_> {}

impl<'a> Interpreter<'a> {
    /// Creates new [`Interpreter`]
    ///
    /// # Arguments
    ///
    /// * `model`: TensorFlow Lite [model][`Model`]
    /// * `options`: Interpreter [options][`Options`]
    ///
    /// # Examples
    ///
    /// ```
    /// use tflitec::model::Model;
    /// use tflitec::interpreter::Interpreter;
    /// let model = Model::new("tests/add.bin")?;
    /// let interpreter = Interpreter::new(&model, None)?;
    /// # Ok::<(), tflitec::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails internally.
    pub fn new(model: &'a Model<'a>, options: Options) -> Result<Interpreter<'a>> {
        unsafe {
            let options_ptr = TfLiteInterpreterOptionsCreate();
            if options_ptr.is_null() {
                return Err(Error::new(ErrorKind::FailedToCreateInterpreter));
            }

            let delegate_ptr: Option<*mut TfLiteDelegate> = match &options {
                    Options::Default => {
                        None
                    },
                    #[cfg(feature = "xnnpack")]
                    Options::Xnnpack(thread_count) => {
                        TfLiteInterpreterOptionsSetNumThreads(options_ptr, *thread_count);
                        Some(Interpreter::configure_xnnpack(options_ptr, *thread_count))
                    },
                    #[cfg(feature = "external_delegate")]
                    Options::External(delegate_path) => {
                        TfLiteInterpreterOptionsSetNumThreads(options_ptr, -1);
                        Some(Interpreter::configure_external_delegate(options_ptr, &delegate_path))
                    }
                };

            // TODO(ebraraktas): TfLiteInterpreterOptionsSetErrorReporter
            let model_ptr = model.model_ptr as *const TfLiteModel;
            let interpreter_ptr = TfLiteInterpreterCreate(model_ptr, options_ptr);
            TfLiteInterpreterOptionsDelete(options_ptr);

            if interpreter_ptr.is_null() {
                Err(Error::new(ErrorKind::FailedToCreateInterpreter))
            } else {
                Ok(Interpreter {
                    options,
                    interpreter_ptr,
                    delegate_ptr,
                    model,
                })
            }
        }
    }

    /// Returns the total number of input [`Tensor`]s associated with the model.
    pub fn input_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetInputTensorCount(self.interpreter_ptr) as usize }
    }

    /// Returns the total number of output `Tensor`s associated with the model.
    pub fn output_tensor_count(&self) -> usize {
        unsafe { TfLiteInterpreterGetOutputTensorCount(self.interpreter_ptr) as usize }
    }

    /// Invokes the interpreter to perform inference from the loaded graph.
    ///
    /// # Errors
    ///
    /// Returns error if TensorFlow Lite C fails to invoke.
    pub fn invoke(&self) -> Result<()> {
        if TfLiteStatus_kTfLiteOk == unsafe { TfLiteInterpreterInvoke(self.interpreter_ptr) } {
            Ok(())
        } else {
            Err(Error::new(ErrorKind::AllocateTensorsRequired))
        }
    }

    /// Returns the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the input [`Tensor`].
    ///
    /// # Errors
    ///
    /// Returns error if [`Interpreter::allocate_tensors()`] was not called before calling this
    /// or given index is not a valid input tensor index in
    /// [0, [`Interpreter::input_tensor_count()`]).
    pub fn input(&self, index: usize) -> Result<Tensor> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut TfLiteTensor).map_err(|error| {
                if error.kind() == ErrorKind::ReadTensorError {
                    Error::new(ErrorKind::AllocateTensorsRequired)
                } else {
                    error
                }
            })
        }
    }

    /// Returns the output [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the output [`Tensor`].
    ///
    /// # Errors
    ///
    /// Returns error if given index is not a valid output tensor index in
    /// [0, [`Interpreter::output_tensor_count()`]). And, it may return error
    /// unless the output tensor has been both sized and allocated. In general,
    /// best practice is to call this *after* calling [`Interpreter::invoke()`].
    pub fn output(&self, index: usize) -> Result<Tensor> {
        let max_index = self.output_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetOutputTensor(self.interpreter_ptr, index as i32);
            Tensor::from_raw(tensor_ptr as *mut TfLiteTensor).map_err(|error| {
                if error.kind() == ErrorKind::ReadTensorError {
                    Error::new(ErrorKind::InvokeInterpreterRequired)
                } else {
                    error
                }
            })
        }
    }

    /// Resizes the input [`Tensor`] at the given index to the
    /// specified [`Shape`][tensor::Shape].
    ///
    /// - Note: After resizing an input tensor, the client **must** explicitly call
    /// [`Interpreter::allocate_tensors()`] before attempting to access the resized tensor data
    /// or invoking the interpreter to perform inference.
    ///
    /// # Arguments
    ///
    /// * `index`: The index for the input [`Tensor`].
    /// * `shape`: The shape to resize the input [`Tensor`] to.
    ///
    /// # Errors
    ///
    /// Returns error if given index is not a valid input tensor index in
    /// [0, [`Interpreter::input_tensor_count()`]) or TensorFlow Lite C fails internally.
    pub fn resize_input(&self, index: usize, shape: tensor::Shape) -> Result<()> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        let dims = shape
            .dimensions()
            .iter()
            .map(|v| *v as i32)
            .collect::<Vec<i32>>();

        unsafe {
            if TfLiteStatus_kTfLiteOk
                == TfLiteInterpreterResizeInputTensor(
                    self.interpreter_ptr,
                    index as i32,
                    dims.as_ptr() as *const c_int,
                    dims.len() as i32,
                )
            {
                Ok(())
            } else {
                Err(Error::new(ErrorKind::FailedToResizeInputTensor(index)))
            }
        }
    }

    /// Allocates memory for all input [`Tensor`]s and dependent tensors based on
    /// their [`Shape`][tensor::Shape]s.
    ///
    /// - Note: This is a relatively expensive operation and should only be called
    /// after creating the interpreter and resizing any input tensors.
    ///
    /// # Error
    ///
    /// Returns error if TensorFlow Lite C fails to allocate memory
    /// for the input tensors.
    pub fn allocate_tensors(&self) -> Result<()> {
        if TfLiteStatus_kTfLiteOk
            == unsafe { TfLiteInterpreterAllocateTensors(self.interpreter_ptr) }
        {
            Ok(())
        } else {
            Err(Error::new(ErrorKind::FailedToAllocateTensors))
        }
    }

    /// Copies the given `data` to the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to be copied to the input `Tensor`'s data buffer
    /// * `index`: The index for the input `Tensor`
    ///
    /// # Errors
    ///
    /// Return error if the data length does not match the buffer size of the input tensor or
    /// the given index is not a valid input tensor index in
    /// [0, [`Interpreter::input_tensor_count()`]) or TensorFlow Lite C fails internally.
    fn copy_bytes(&self, data: &[u8], index: usize) -> Result<()> {
        let max_index = self.input_tensor_count() - 1;
        if index > max_index {
            return Err(Error::new(ErrorKind::InvalidTensorIndex(index, max_index)));
        }
        unsafe {
            let tensor_ptr = TfLiteInterpreterGetInputTensor(self.interpreter_ptr, index as i32);
            let byte_count = TfLiteTensorByteSize(tensor_ptr);
            if data.len() != byte_count {
                return Err(Error::new(ErrorKind::InvalidTensorDataCount(
                    data.len(),
                    byte_count,
                )));
            }
            let status =
                TfLiteTensorCopyFromBuffer(tensor_ptr, data.as_ptr() as *const c_void, data.len());
            if status != TfLiteStatus_kTfLiteOk {
                Err(Error::new(ErrorKind::FailedToCopyDataToInputTensor))
            } else {
                Ok(())
            }
        }
    }

    /// Copies the given `data` to the input [`Tensor`] at the given `index`.
    ///
    /// # Arguments
    ///
    /// * `data`: The data to be copied to the input `Tensor`'s data buffer.
    /// * `index`: The index for the input [`Tensor`].
    ///
    /// # Errors
    ///
    /// Returns error if byte count of the data does not match the buffer size of the
    /// input tensor or the given index is not a valid input tensor index in
    /// [0, [`Interpreter::input_tensor_count()`]) or TensorFlow Lite C fails internally.
    pub fn copy<T>(&self, data: &[T], index: usize) -> Result<()> {
        let element_size = std::mem::size_of::<T>();
        let d = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * element_size)
        };
        self.copy_bytes(d, index)
    }

    /// Returns optional reference of [`Options`].
    pub fn options(&self) -> &Options {
        &self.options
    }

    #[cfg(feature = "xnnpack")]
    unsafe fn configure_xnnpack(
        interpreter_options_ptr: *mut TfLiteInterpreterOptions,
        thread_count: i32,
    ) -> *mut TfLiteDelegate {

        let mut xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
        if thread_count > 0 {
            xnnpack_options.num_threads = thread_count;
        }

        let xnnpack_delegate_ptr = TfLiteXNNPackDelegateCreate(&xnnpack_options);
        TfLiteInterpreterOptionsAddDelegate(interpreter_options_ptr, xnnpack_delegate_ptr);
        xnnpack_delegate_ptr
    }

    #[cfg(feature = "external_delegate")]
    unsafe fn configure_external_delegate(
        interpreter_options_ptr: *mut TfLiteInterpreterOptions,
        external_delegate_path: &str,
    ) -> *mut TfLiteDelegate {
        let c_delegate_path = CString::new(external_delegate_path).unwrap();
        let external_delegate_options =
            TfLiteExternalDelegateOptionsDefault(c_delegate_path.as_ptr());
        let external_delegate_ptr = TfLiteExternalDelegateCreate(&external_delegate_options);
        TfLiteInterpreterOptionsAddDelegate(interpreter_options_ptr, external_delegate_ptr);
        external_delegate_ptr
    }


}

impl Drop for Interpreter<'_> {
    fn drop(&mut self) {
        unsafe {
            TfLiteInterpreterDelete(self.interpreter_ptr);
            {
                match &self.options {
                    Options::Default => {},
                    #[cfg(feature = "xnnpack")]
                    Options::Xnnpack(_) => {
                        if let Some(delegate_ptr) = self.delegate_ptr{
                            TfLiteXNNPackDelegateDelete(delegate_ptr)
                        }
                    },
                    #[cfg(feature = "external_delegate")]
                    Options::External(_) => {
                        if let Some(delegate_ptr) = self.delegate_ptr{
                            TfLiteExternalDelegateDelete(delegate_ptr)
                        }
                    },
                }

            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use crate::interpreter::Interpreter;
    use crate::interpreter::Options;
    use crate::model::Model;
    use crate::tensor;
    use crate::ErrorKind;

    const MODEL_PATH: &str = "tests/add.tflite";
    const EDGE_MODEL_PATH: &str = "tests/mobilenet_edge.tflite";

    #[test]
    fn test_interpreter_input_output_count() {
        let bytes = std::fs::read(MODEL_PATH).expect("Cannot read model data!");
        let model = Model::from_bytes(&bytes).expect("Cannot load model from bytes");
        let interpreter =
            Interpreter::new(&model, Options::Default).expect("Cannot create interpreter");
        assert_eq!(interpreter.input_tensor_count(), 1);
        assert_eq!(interpreter.output_tensor_count(), 1);
    }

    #[test]
    fn test_interpreter_get_input_tensor() {
        let bytes = std::fs::read(MODEL_PATH).expect("Cannot read model data!");
        let model = Model::from_bytes(&bytes).expect("Cannot load model from bytes!");
        let interpreter =
            Interpreter::new(&model, Options::Default).expect("Cannot create interpreter!");

        let invalid_tensor = interpreter.input(1);
        assert!(invalid_tensor.is_err());
        let err = invalid_tensor.err().unwrap();
        assert_eq!(ErrorKind::InvalidTensorIndex(1, 0), err.kind());

        let invalid_tensor = interpreter.input(0);
        assert!(invalid_tensor.is_err());
        let err = invalid_tensor.err().unwrap();
        assert_eq!(ErrorKind::AllocateTensorsRequired, err.kind());

        interpreter.allocate_tensors().unwrap();
        let valid_tensor = interpreter.input(0);
        assert!(valid_tensor.is_ok());
        let tensor = valid_tensor.ok().unwrap();
        assert_eq!(tensor.shape().dimensions(), &vec![1, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_allocate_tensors() {
        let bytes = std::fs::read(MODEL_PATH).expect("Cannot read model data!");
        let model = Model::from_bytes(&bytes).expect("Cannot load model from bytes!");
        let interpreter =
            Interpreter::new(&model, Options::Default).expect("Cannot create interpreter!");

        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.input(0).unwrap();
        assert_eq!(tensor.shape().dimensions(), &vec![10, 8, 8, 3])
    }

    #[test]
    fn test_interpreter_copy_input() {
        let bytes = std::fs::read(MODEL_PATH).expect("Cannot read model data!");
        let model = Model::from_bytes(&bytes).expect("Cannot load model from bytes!");
        let interpreter =
            Interpreter::new(&model, Options::Default).expect("Cannot create interpreter!");

        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let tensor = interpreter.input(0).unwrap();
        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert_eq!(data, tensor.data());
    }

    #[test]
    fn test_interpreter_invoke() {
        let model = Model::new(Path::new(MODEL_PATH)).expect("Cannot load model from file!");
        let interpreter =
            Interpreter::new(&model, Options::Default).expect("Cannot create interpreter!");

        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");

        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert!(interpreter.invoke().is_ok());
        let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
        let output_tensor = interpreter.output(0).unwrap();
        assert_eq!(output_tensor.shape().dimensions(), &vec![10, 8, 8, 3]);
        let output_vector = output_tensor.data::<f32>().to_vec();
        assert_eq!(expected, output_vector);
    }

    #[cfg(feature = "xnnpack")]
    #[test]
    fn test_interpreter_invoke_xnnpack() {
        use crate::interpreter::Options;
        let options = Options::Xnnpack(2);
        let model = Model::new(Path::new(MODEL_PATH)).expect("Cannot load model from file!");
        let interpreter = Interpreter::new(&model, options).expect("Cannot create interpreter!");

        interpreter
            .resize_input(0, tensor::Shape::new(vec![10, 8, 8, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");

        let data = (0..1920).map(|x| x as f32).collect::<Vec<f32>>();
        assert!(interpreter.copy(&data[..], 0).is_ok());
        assert!(interpreter.invoke().is_ok());
        let expected: Vec<f32> = data.iter().map(|e| e * 3.0).collect();
        let output_tensor = interpreter.output(0).unwrap();
        assert_eq!(output_tensor.shape().dimensions(), &vec![10, 8, 8, 3]);
        let output_vector = output_tensor.data::<f32>().to_vec();
        assert_eq!(expected, output_vector);
    }
    
    #[cfg(feature = "external_delegate")]
    #[test]
    fn test_interpreter_invoke_edge_tpu() {
        use crate::interpreter::Options;
        let options = Options::External("libedgetpu.so.1".to_string());
        let model = Model::new(Path::new(EDGE_MODEL_PATH)).expect("Cannot load model from file!");
        let interpreter = Interpreter::new(&model, options).expect("Cannot create interpreter");
        interpreter
            .resize_input(0, tensor::Shape::new(vec![1, 224, 224, 3]))
            .expect("Resize failed");
        interpreter
            .allocate_tensors()
            .expect("Cannot allocate tensors");
        let data = [1u8; 150528];
        assert!(interpreter.copy(&data, 0).is_ok());
        assert!(interpreter.invoke().is_ok());
    }
}
