from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return array_api.log(
            array_api.sum(array_api.exp(Z-array_api.max(Z, axis=self.axes, keepdims=True)), 
                          axis=self.axes)) + array_api.max(Z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(node.inputs[0].realize_cached_data(), 
                              axis=self.axes, 
                              keepdims=True) #! 防止数值溢出
        exp_z = exp(node.inputs[0]-max_z)
        sum_exp_z = exp_z.sum(axes=self.axes)

        if self.axes is None:
            return out_grad.broadcast_to(node.inputs[0].shape) * exp_z / sum_exp_z
        new_shape = list(node.inputs[0].shape)
        if isinstance(self.axes, int):
            self.axes = (self.axes,)
        for axis in self.axes:
            new_shape[axis] = 1
        out_grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        sum_exp_z = sum_exp_z.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        return out_grad * exp_z / sum_exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

