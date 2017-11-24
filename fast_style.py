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

import os
import datetime
import mxnet as mx
import numpy as np
import importlib
import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='fast style')

    parser.add_argument('--model', type=str, default='vgg19',
                        choices = ['vgg'],
                        help = 'the pretrained model to use')
    parser.add_argument('--train', type=int, default=0,
                        help='train or transfer')
    parser.add_argument('--input_image', type=str, default='input/IMG_4343.jpg',
                        help='the content image')
    parser.add_argument('--style_image', type=str, default='datas/style_images/starry_night.jpg',
                        help='the style image')
    parser.add_argument('--train_path', type=str, default='datas/training_images/',
                        help='the style image')
    parser.add_argument('--content_weight', type=float, default=15,
                        help='the weight for the content image')
    parser.add_argument('--style_weight', type=float, default=2,
                        help='the weight for the style image')
    parser.add_argument('--tv_weight', type=float, default=1e-2,
                        help='the magtitute on TV loss')
    parser.add_argument('--max_num_epochs', type=int, default=1,
                        help='the maximal number of training epochs')
    parser.add_argument('--imagesize', type=int, default=256,
                        help='resize the content image')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='the initial learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu card to use, -1 means using cpu')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='the output image')
    parser.add_argument('--remove_noise', type=float, default=.02,
                        help='the magtitute to remove noise')
    parser.add_argument('--lr_sched_delay', type=int, default=75,
                        help='how many epochs between decreasing learning rate')
    parser.add_argument('--lr_sched_factor', type=int, default=0.9,
                        help='factor to decrease learning rate on schedule')

    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)


def PreprocessImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename, remove_noise=0.):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)

def style_gram_symbol(input_size, style):
    _, output_shapes, _ = style.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], target_shape=(int(shape[1]), int(np.prod(shape[2:]))))
        # use fully connected to quickly do dot(x, x^T)
        gram = mx.sym.FullyConnected(x, x, no_bias=True, num_hidden=shape[1])
        gram_list.append(gram)
        grad_scale.append(np.prod(shape[1:]) * shape[1])
    return mx.sym.Group(gram_list), grad_scale


def get_loss(gram, content):
    gram_loss = []
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(gram_loss), content_loss

def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})

def train_fstyle(args):
    """Train a neural style network.
    Args are from argparse and control input, output, hyper-parameters.
    """
    # input
    dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    data_shape = (1, 3, args.imagesize, args.imagesize)
    style_np = PreprocessImage(args.style_image, data_shape)
    size = data_shape[2:]

    # loss model
    model_module =  importlib.import_module('model_' + args.model)
    style, content = model_module.get_symbol()
    gram, gscale = style_gram_symbol(size, style)
    feature_executor = model_module.get_executor(gram, content, size, dev)
    feature_executor.data[:] = style_np
    feature_executor.executor.forward()
    style_array = []
    for i in range(len(feature_executor.style)):
        style_array.append(feature_executor.style[i].copyto(mx.cpu()))

    style_loss, content_loss = get_loss(gram, content)
    loss_executor = model_module.get_executor(
        style_loss, content_loss, size, dev)

    grad_array = []
    for i in range(len(style_array)):
        style_array[i].copyto(loss_executor.arg_dict["target_gram_%d" % i])
        grad_array.append(mx.nd.ones((1,), dev) * (float(args.style_weight) / gscale[i]))
    grad_array.append(mx.nd.ones((1,), dev) * (float(args.content_weight)))

    print([x.asscalar() for x in grad_array])

    # fast model
    fast_module =  importlib.import_module('model_transfer')
    fast_executor =  fast_module.fast_style_model('fast', dev, size) 

    # train
    #lr = mx.lr_scheduler.FactorScheduler(step=args.lr_sched_delay,
    #        factor=args.lr_sched_factor)

    optimizer = mx.optimizer.Adam(
        learning_rate = args.lr,
        wd = 0.005)
        #momentum=0.95,
        #lr_scheduler = lr)

    optim_state = []
    for i, item in enumerate(fast_executor.arg_names):
        optim_state.append(optimizer.create_state(i, fast_executor.arg_dict[item]))

    clip_norm = 0.05 * np.prod(data_shape)

    # construct a callback function to save checkpoints
    style_name = os.path.splitext(os.path.basename(args.style_image))[0]
    model_prefix = args.output_dir + style_name  + '/' + style_name
    if not os.path.exists(args.output_dir + style_name):
        os.makedirs(args.output_dir + style_name)
    saveParamsCallback = mx.callback.do_checkpoint(model_prefix)

    logging.info('start training arguments %s', args)
    for e in range(args.max_num_epochs):
        for fname in sorted(os.listdir(args.train_path)):
            train_file = os.path.join(args.train_path, fname)
            content_np = PreprocessImage(train_file, data_shape)

            feature_executor.data[:] = content_np
            feature_executor.executor.forward()
            content_array = feature_executor.content.copyto(mx.cpu())
            content_array.copyto(loss_executor.arg_dict["target_content"])

            fast_executor.data[:] = content_np
            fast_executor.executor.forward(is_train=True)

            loss_executor.data[:] = fast_executor.fast_out
            loss_executor.executor.forward(is_train=True)
            loss_executor.executor.backward(grad_array)

            tv_grad_executor = get_tv_grad_executor(fast_executor.fast_out, dev, args.tv_weight)
            tv_grad_executor.forward()
            grad = loss_executor.data_grad + tv_grad_executor.outputs[0]
            gnorm = mx.nd.norm(grad).asscalar()
            if gnorm > clip_norm:
                grad[:] *= clip_norm / gnorm
            fast_executor.executor.backward(grad)
            for i, item in enumerate(fast_executor.arg_names):
                optimizer.update(i, fast_executor.arg_dict[item], fast_executor.grad_dict[item], optim_state[i])

        saveParamsCallback(e, fast_executor.sym, fast_executor.arg_dict, fast_executor.aux_dict)
        print('epoch %d:', e)
        print(loss_executor.style)
        print(loss_executor.content)

    old_time=datetime.datetime.now()
    content_np = PreprocessImage(args.input_image, data_shape)
    fast_executor.data[:] = content_np
    fast_executor.executor.forward()
    new_time=datetime.datetime.now()
    print 'cost: %s us' % (new_time-old_time).microseconds

    old_Img_name = os.path.basename(args.input_image)
    new_Img_name = args.output_dir + style_name + '_' + old_Img_name
    new_Img = fast_executor.fast_out.asnumpy()
    SaveImage(new_Img, new_Img_name, args.remove_noise)

def fstyle(args):
    dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    temp_img = io.imread(args.input_image)
    data_shape = (1, 3, temp_img.shape[0], temp_img.shape[1])
    del temp_img
    size = data_shape[2:]

    # fast model
    fast_module =  importlib.import_module('model_transfer')
    fast_executor =  fast_module.fast_style_model('fast', dev, size)
    # load net and params
    style_name = os.path.splitext(os.path.basename(args.style_image))[0]
    model_prefix = args.output_dir + style_name  + '/' + style_name
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.max_num_epochs)
    #assert sym.tojson() == fast_executor.sym.tojson()
    for key in arg_params:
        if key in fast_executor.arg_names:
            arg_params[key].copyto(fast_executor.arg_dict[key])
    for key in aux_params:
        if fast_executor.aux_dict.has_key(key):
            aux_params[key].copyto(fast_executor.aux_dict[key])
    del sym
    del arg_params
    del aux_params

    old_time=datetime.datetime.now()
    content_np = PreprocessImage(args.input_image, data_shape)
    fast_executor.data[:] = content_np
    del content_np
    fast_executor.executor.forward()
    new_time=datetime.datetime.now()
    print 'cost: %s us' % (new_time-old_time).microseconds

    old_Img_name = os.path.basename(args.input_image)
    new_Img_name = args.output_dir + style_name + '_' + old_Img_name
    new_Img = fast_executor.fast_out.asnumpy()
    SaveImage(new_Img, new_Img_name, args.remove_noise)

if __name__ == "__main__":
    args = get_args()
    if args.train == 1:
        print 'train...'
        train_fstyle(args)
    elif args.train == 0:
        print 'transfer...'
        fstyle(args)

