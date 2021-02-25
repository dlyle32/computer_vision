import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import random
import math
import datetime
import time
import os
import argparse
import logging
from tensorflow.keras.utils import plot_model
import argparse

logger = logging.getLogger("comp_vis")
# tf.compat.v1.disable_eager_execution()

def main(args):

    timestamp = int(time.time())
    logdir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), args.logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    hdlr = logging.FileHandler(os.path.join(logdir, "training_output_%d.log" % timestamp))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    checkpointdir = os.path.join(args.volumedir, datetime.datetime.today().strftime('%Y%m%d'), args.checkpointdir)
    if not os.path.isdir(checkpointdir):
        os.makedirs(checkpointdir)
    
  # Dynamically load modelBuilder class
    moduleName, klassName = args.modelbuilder.split(".")
    mod = __import__('models.%s' % moduleName, fromlist=[klassName])
    klass = getattr(mod,klassName)
    modelBuilder = klass(args)

    optimizer_map = {"adam": Adam, "rmsprop": RMSprop, "sgd": SGD}
    optimizer = optimizer_map[args.optimizer] if args.optimizer in optimizer_map.keys() else RMSprop

    lr_decay = ExponentialDecay(initial_learning_rate=args.learningrate,
                                decay_rate=args.decayrate,
                                decay_steps=args.decaysteps)

    opt = optimizer(learning_rate=lr_decay, clipvalue=3)

    # model = modelBuilder.get_model()
    # # model = modelBuilder.compile_model(model, opt)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # print(model.summary())
    C,S,G  = modelBuilder.get_train_data()
    for i in range(1, args.numepochs+1):
        loss, grads = modelBuilder.compute_loss_and_grads(C,S,G)
        opt.apply_gradients([(grads, G)])
        if i % 100 == 0:
            logger.info("Iteration: %d, Loss: %.2f" % (i,loss))
            modelBuilder.save_generated_img(G.numpy(), i)
    # callbacks = modelBuilder.get_callbacks(model)
    # init_epoch = 0
    # history = model.fit(X, Y,
    #            epochs=args.numepochs,
    #            steps_per_epoch=1,
    #            initial_epoch=init_epoch,
    #            callbacks=[])
    # modelBuilder.save_generated_img(G.numpy(), args.numepochs)
    # logger.info(history.history)


def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--loadmodel", type=str)
    parser.add_argument("--datadir", default="data/")
    parser.add_argument("--logdir", default="logs/")
    parser.add_argument("--volumedir", default="/training/")
    parser.add_argument("--checkpointdir", default="checkpoints/")
    parser.add_argument("--checkpointnames", default="nodle_char_model.%d.{epoch:03d}.h5")
    parser.add_argument("--minibatchsize", type=int, default=256)
    parser.add_argument("--numepochs", type=int, default=25)
    parser.add_argument("--learningrate", type=float, default=0.01)
    parser.add_argument("--dropoutrate", type=float, default=0.2)
    parser.add_argument("--regfactor", type=float, default=0.01)
    parser.add_argument("--modelbuilder", type=str)
    parser.add_argument("--valsplit", type=float, default=0.2)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--decaysteps", type=int, default=10000)
    parser.add_argument("--decayrate", type=float, default=1.0)
    parser.add_argument("--content_factor", type=float, default=0.5)
    parser.add_argument("--style_factor", type=float, default=0.5)
    parser.add_argument("--content_img", type=str, default="data/content.jpg")
    parser.add_argument("--style_img", type=str, default="data/style.jpg")
    parser.add_argument("--imgwidth", type=int, default=224)
    parser.add_argument("--imgheight", type=int, default=224)

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    main(args)
