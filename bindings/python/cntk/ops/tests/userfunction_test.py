# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for function extension
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import AA, I, precision, PRECISION_TO_TYPE, compare_lists_of_np_arrays
from ...utils import sanitize_dtype_cntk, eval as cntk_eval, cntk_device

from cntk import *
from cntk.trainer import *
from cntk.learner import *
from cntk.ops.functions import Function, UserFunction, UserOutput

class Plus3Func(UserFunction):

    def __init__(self, in1):
        outputs = [UserOutput(in1.shape, in1.dtype, in1.dynamic_axes)]
        super(Plus3Func, self).__init__([in1], outputs, op_name='Plus3Func', name='f1')

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        assert len(arguments)==1
        for v in arguments.values():
            break

        assert len(outputs)==1
        for k in outputs:
            outputs[k] = v + 3

    def backward(self, state, root_gradients, variables):
        assert len(root_gradients) == 1
        assert len(variables) == 1

        for rk, rv in root_gradients.items():
            break
        for var_key in variables:
            break

        variables[var_key] = rv

def test_ext_eval(precision):
    dim = 4

    p = parameter(shape=(dim,), init=10)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = Plus3Func(i)
    z = m+p

    input_data = np.random.rand(dim)
    result = z.eval([input_data])
    assert np.allclose(result[0][0]-input_data, 13+np.zeros_like(input_data))

# TODO change to real training example
def test_ext_train(precision):
    dim = 4

    p = parameter(shape=(dim,), init=10)
    i = input_variable(dim, needs_gradient=True, name='i_var')
    m = Plus3Func(i)
    z = m+p

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, z+0, z+0, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant)])

    i = 0
    while i<100:
        i+=1
        input_data = np.random.rand(dim)
        trainer.train_minibatch([input_data])
