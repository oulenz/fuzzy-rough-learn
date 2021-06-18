"""Neural network layers"""

from tensorflow.keras import activations, constraints, initializers, layers, regularizers
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes, tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops


class TransposedDense(layers.Layer):

    def __init__(self,
                 dense,
                 units,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.dense = dense
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `TransposedDense` layer with non-floating point '
                          'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `TransposedDense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        if self.use_bias:
            self.bias = self.add_weight(
              'bias',
              shape=[self.units, ],
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
            self.bias = None
        super().build(input_shape)
        self.built = True

    def call(self, inputs):
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.dense.kernel, [[rank - 1], [1]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, self.dense.kernel, adjoint_b=True)
            else:
                outputs = gen_math_ops.mat_mul(inputs, self.dense.kernel, transpose_b=True)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
