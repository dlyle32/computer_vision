import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.ops import special_math_ops
from tensorflow.keras.callbacks import LambdaCallback
import os
from PIL import Image
import cv2
import numpy as np
import logging

# Neural style transfer model
# Generates an image by iteratively improving the style and content
# based on a loss function that compares the generated image
# to an image meant to represent the style and another meant to represent the content

logger = logging.getLogger("comp_vis")
def compute_style_matrix(G):
    op = "ijk,ijl->kl"
    correlation_layer = EinsumOp(op)
    return correlation_layer([G,G])

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


class EinsumOp(keras.layers.Layer):
    def __init__(self, op, **kwargs):
        super(EinsumOp, self).__init__(**kwargs)
        self.op = op

    def call(self, inputs):
        a1 = inputs[0]
        a2 = inputs[1]
        attn_factor = special_math_ops.einsum(self.op, a1, a2)
        return attn_factor

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "op" : self.op
        })
        return config

def sum_square_diff(A,B):
    square_diff = tf.square(A-B)
    return tf.reduce_sum(square_diff)

def add_batch_dim(A):
    return A[np.newaxis, :, : :]

def read_and_resize_img(path, nh, nw):
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(nw, nh), interpolation=cv2.INTER_AREA)
    img = tf.constant(img, dtype=tf.float32)
    return img

class ImageGen(keras.layers.Layer):
    def __init__(self, nh, nw, nc, **kwargs):
        super(ImageGen,self).__init__(**kwargs)
        self.nh = nh
        self.nw = nw
        self.nc = nc
        self.w = self.add_weight(
            shape=(1,nh,nw,nc),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return self.w

    def get_img_matrix(self): 
        weights = self.get_weights()[0]
        return np.reshape(weights, (self.nh,self.nw,self.nc))

class StyleTransferBuilder():

    def __init__(self, args):
        self.content_factor = args.content_factor
        self.style_factor = args.style_factor
        self.style_layers = ["block2_conv1", "block3_conv3", "block4_conv2", "block5_conv1"]
        self.content_layer = "block5_conv2"
        self.style_layer_weights = [0.25] * 4
        self.datadir = args.datadir
        self.contentpath = args.content_img
        self.stylepath = args.style_img
        self.nh = args.imgheight
        self.nw = args.imgwidth
        self.nc = 3
        self.vgg = VGG16(include_top=False)

    def loss(self, model, content_img, style_img, gen_img):
        concat_input = tf.concat([content_img, style_img, gen_img], axis=0)
        outs = model(concat_input)
        content_features = outs[self.content_layer]
        C = content_features[0, :, :, :]
        G = content_features[2, :, :, :]
        loss = tf.zeros(shape=())
        loss = loss + self.content_factor * self.content_loss(C, G)
        for i,l in enumerate(self.style_layers):
            feats = outs[l]
            S = feats[1, :, :, :]
            G = feats[2, :, :, :]
            loss = loss + self.style_factor * self.style_layer_weights[i] * self.style_loss(S, G)

        return loss

    @tf.function
    def compute_loss_and_grads(self, model, content_img, style_img, gen_img):
        with tf.GradientTape() as tape:
            tape.watch(gen_img)
            loss = self.loss(model, content_img, style_img, gen_img)
        grads = tape.gradient(loss, gen_img)
        return loss, grads

    def content_loss(self, C, G):
        return sum_square_diff(C, G)

    def style_loss(self, S, G):
        # S = compute_style_matrix(S)
        S = gram_matrix(S)
        # G = compute_style_matrix(G)
        G = gram_matrix(G)
        return sum_square_diff(S,G)

    def get_model(self):
        outputs_dict = dict([(layer.name, layer.output) for layer in self.vgg.layers])
        model = keras.Model(inputs=self.vgg.inputs, outputs=outputs_dict)
        return model


    def get_vgg_model(self):
        block4_conv2 = self.vgg.get_layer(self.content_layer).output
        outputs = {self.content_layer: block4_conv2}
        for i, l in enumerate(self.style_layers):
            sl = self.vgg.get_layer(l).output
            # style_correlation = compute_style_matrix(sl, i)
            # outputs.append(style_correlation)
            outputs[l] = sl
        model = keras.Model(inputs=self.vgg.input, outputs=outputs)
        for layer in model.layers:
            layer.trainable = False
        return model

    
    def precompute_content_style_vals(self, model, C, S):
        concat_input = tf.concat([C,S], axis=0)
        # yc = model(C[np.newaxis,:,:,:])
        # ys = model(S[np.newaxis,:,:,:])
        y = model(concat_input)
        return y

    def get_train_data(self):
        content_img = read_and_resize_img(self.contentpath, self.nh, self.nw)
        style_img = read_and_resize_img(self.stylepath, self.nh, self.nw)
        img = cv2.imread(self.contentpath)
        img = cv2.resize(img, dsize=(self.nw, self.nh), interpolation=cv2.INTER_AREA)
        gen_img = tf.Variable(img[np.newaxis, :,:,:], dtype=tf.float32, trainable=True)
        # gen_img = tf.Variable(tf.random.normal((1,content_img.shape[0], content_img.shape[1], content_img.shape[2])), dtype=tf.float32, trainable=True)
        # Y = self.precompute_content_style_vals(content_arr, style_arr)
        content_img = add_batch_dim(content_img)
        style_img = add_batch_dim(style_img)
        return content_img, style_img, gen_img

    def save_generated_img(self, snapdir, image_matrix, epoch, timestamp):
        image_matrix = image_matrix[0,:,:,:]
        img_path = os.path.join(snapdir, "gen_image_%d_%d.jpg" % (timestamp,epoch))
        cv2.imwrite(img_path, image_matrix)

    def get_callbacks(self, model):
        save_gen_img = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_generated_img(model, epoch))
        return [save_gen_img]




        





