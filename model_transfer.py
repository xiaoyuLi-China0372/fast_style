# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from collections import namedtuple

FastExecutor = namedtuple('FastExecutor', ['executor', 'data', 'arg_names', 'grad_dict', 'fast_out', 'arg_dict', 'aux_dict', 'sym'])

def DeconvFactory(data, target_shape, num_filter, kernel, stride=(1, 1), act_type="relu", mirror_attr={}, name=None, suffix='', with_act=True):
    deconv = mx.sym.Deconvolution(data=data, target_shape=target_shape, num_filter=num_filter, kernel=kernel, stride=stride, no_bias=True, name='deconv_%s%s' %(name, suffix))
    bn = mx.sym.BatchNorm(data=deconv, name='bn_%s%s' %(name, suffix))
    if with_act:
        act = mx.sym.Activation(data=bn, act_type=act_type, attr=mirror_attr, name='relu_%s%s' %(name, suffix))
        return act
    else:
        return bn

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), act_type="relu", mirror_attr={}, name=None, suffix='', with_act=True):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    if with_act:
        act = mx.sym.Activation(data=bn, act_type=act_type, attr=mirror_attr, name='relu_%s%s' %(name, suffix))
        return act
    else:
        return bn

def ResBlock(data, num_filter, name):
    conv1 = ConvFactory(data, num_filter, (3, 3), name=name, suffix='_1')
    conv2 = ConvFactory(conv1, num_filter, (3, 3), name=name, suffix='_2', with_act=False)
    data = data + conv2
    return data


def fast_style_net(prefix, imHw=(256,256)):
    data = mx.sym.Variable(prefix + '_data')
    conv1 = ConvFactory(data, num_filter=32, kernel=(9,9), stride=(1,1), pad=(4,4), name='1', suffix='_1')
    conv2 = ConvFactory(conv1, num_filter=64, kernel=(3,3), stride=(2,2), pad=(1,1), name='2', suffix='_2')
    conv3 = ConvFactory(conv2, num_filter=128, kernel=(3,3), stride=(2,2), pad=(1,1), name='3', suffix='_3')
    res1 = ResBlock(conv3, num_filter=128, name='4')
    res2 = ResBlock(res1, num_filter=128, name='5')
    res3 = ResBlock(res2, num_filter=128, name='6')
    res4 = ResBlock(res3, num_filter=128, name='7')
    res5 = ResBlock(res4, num_filter=128, name='8')
    deConv1 = DeconvFactory(res5, target_shape=(imHw[0]/2, imHw[1]/2), num_filter=64, kernel=(4,4), stride=(2,2), name='9', suffix='_1')
    deConv2 = DeconvFactory(deConv1, target_shape=(imHw[0], imHw[1]), num_filter=32, kernel=(3,3), stride=(2,2), name='10', suffix='_2')
    conv4 = mx.sym.Convolution(deConv2, num_filter=3, kernel=(9,9), stride=(1,1), pad=(4,4), name='conv_11')
    act4 = mx.sym.Activation(conv4, act_type='tanh', name='relu_11')
    rawOut = (act4 * 128) + 128
    norm = mx.sym.SliceChannel(rawOut, num_outputs=3)
    rCh = norm[0] - 123.68
    gCh = norm[1] - 116.779
    bCh = norm[2] - 103.939
    normOut = mx.sym.Concat(rCh, gCh, bCh, name=prefix+'_out')
    return normOut

def get_executor(prefix, net, input_size, ctx):
    out = net
    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(fast_data=(1, 3, input_size[0], input_size[1]))
    arg_names = out.list_arguments()
    param_names = [x for x in arg_names if x != prefix + '_data']
    aux_names = out.list_auxiliary_states()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=ctx) for shape in aux_shapes]))
    arg_shapes.remove((1, 3, input_size[0], input_size[1]))
    grad_dict = dict(zip(param_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    initializer = mx.init.Xavier(magnitude=2)
    for name in param_names:
        initializer(mx.init.InitDesc(name), arg_dict[name]) 
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write", aux_states=aux_dict)
    return FastExecutor(executor=executor,
                        data=arg_dict[prefix + '_data'],
                        arg_names=param_names,
                        grad_dict=grad_dict,
                        fast_out=executor.outputs[0],
                        arg_dict=arg_dict,
                        aux_dict=aux_dict,
                        sym=out)


def fast_style_model(prefix, context, data_shape=(256,256)):
    net = fast_style_net(prefix, data_shape)
    return get_executor(prefix, net, data_shape, context)


if __name__ == "__main__":
    net = fast_style_net('fast')
    arg_names = net.list_arguments()
    print arg_names
    #mx.viz.plot_network(net).view()

