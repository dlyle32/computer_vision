import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.ops import special_math_ops
from tensorflow.keras.callbacks import LambdaCallback
import os
from PIL import Image
import numpy as np
import logging

# Neural style transfer model
# Generates an image by iteratively improving the style and content
# based on a loss function that compares the generated image
# to an image meant to represent the style and another meant to represent the content

def compute_style_matrix(G, layer):
    op = "aijk,aijl->kl"
    correlation_layer = EinsumOp(op, name="compute_style_%d" % layer)
    return correlation_layer([G,G])


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
    square_diff = tf.math.square(tf.math.subtract(A, B))
    return tf.math.reduce_sum(square_diff)

class ImageGen(keras.layers.Layer):
    def __init__(self, nh, nw, nc, **kwargs):
        super(ImageGen,self).__init__(kwargs)
        self.w = self.add_weight(
            shape=(nh,nw,nc),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return self.w


class StyleTransferBuilder():

    def __init__(self, args):
        self.content_factor = args.content_factor
        self.style_factor = args.style_factor
        self.style_layers = ["block2_conv1", "block3_conv3", "block4_conv2", "block5_conv1"]
        self.content_layer = "block4_conv2"
        self.style_layer_weights = [0.25] * 4
        self.datadir = args.datadir
        self.nh = 224
        self.nw = 224
        self.nc = 3
        self.vgg = VGG16()

    def loss(self, y, y_hat):
        C = y_hat[0]
        S = y_hat[1:]
        cl = self.content_loss(y[0],C)
        sl = self.style_loss(y[1:], S)
        return self.content_factor * cl + self.style_factor * sl

    def get_model(self):
        inpt = keras.layers.Input(shape=(1), name="input")
        img = ImageGen(self.nh, self.nw, self.nc, name="GenImage")(inpt)
        out = self.vgg.get_layer("block1_conv1")(img[np.newaxis,:,:,:])
        outputs = []
        style_l = 0
        style_out = []
        for layer in self.vgg.layers[2:]:
            print(layer.name)
            out = layer(out)
            if layer.name == self.content_layer:
                outputs.append(out)
            if style_l < len(self.style_layers) and layer.name == self.style_layers[style_l]:
                style_correlation = compute_style_matrix(out, style_l)
                style_out.append(style_correlation)
                style_l += 1
            layer.trainable = False
        outputs.extend(style_out)
        # block4_conv2 = vgg.get_layer(self.content_layer).output
        # outputs = [block4_conv2]
        # for i,l in enumerate(self.style_layers):
        #     sl = vgg.get_layer(l).output
        #     style_correlation = compute_style_matrix(sl, i)
        #     outputs.append(style_correlation)
        model = keras.Model(inputs=inpt, outputs=outputs)
        return model

    def precompute_content_style_vals(self, C, S):
        block4_conv2 = self.vgg.get_layer(self.content_layer).output
        outputs = [block4_conv2]
        for i, l in enumerate(self.style_layers):
            sl = self.vgg.get_layer(l).output
            style_correlation = compute_style_matrix(sl, i)
            outputs.append(style_correlation)
        print("HERE")
        model = keras.Model(inputs=self.vgg.input, outputs=outputs)
        print("HEERE")
        yc = model.predict(C[np.newaxis,:,:,:])[0]
        print("HEEERE")
        ys = model.predict(S[np.newaxis,:,:,:])[1:]
        print("HEEEERE")
        return [yc] + ys

    def get_train_data(self):
        content_img_file = os.path.join(self.datadir, "content.jpg")
        style_img_file = os.path.join(self.datadir, "style.jpg")
        content_img = load_img(content_img_file, target_size=(self.nh,self.nw))
        style_img = load_img(style_img_file, target_size=(self.nh,self.nw))
        content_arr = img_to_array(content_img)
        style_arr = img_to_array(style_img)
        Y = self.precompute_content_style_vals(content_arr, style_arr)
        return [], Y

    def content_loss(self,G, C):
        return sum_square_diff(G,C)

    def style_loss(self, G, S):
        loss = 0
        for i in range(G.shape[0]):
            loss += self.style_layer_weights[i] * sum_square_diff(G[i],S)

    def compile_model(self, model, optimizer):
        model.compile(loss=lambda y,yh : self.content_loss(y,yh), optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    def save_generated_img(self, model, epoch):
        imageGenLayer = model.get_layer("GenImage")
        imageMatrix = imageGenLayer.get_weights()
        img = Image.fromarray(imageMatrix, 'RGB')
        img_path = os.path.join(self.datadir, "gen_image_%d.jpg" % epoch)
        img.save(img_path)

    def get_callbacks(self, model):
        save_gen_img = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_gen_img(model, epoch))
        return [save_gen_img]




        





