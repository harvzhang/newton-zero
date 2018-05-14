import hashlib
import json
import os
from logging import getLogger
# noinspection PyPep8Naming
import keras.backend as K

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
from keras.regularizers import l2

from newton_zero.config import Config

import keras

logger = getLogger(__name__)


class Connect4Model:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None

    def build(self):
        mc = self.config.model
        in_x = x = Input((2, 8, 5))  # [own(8x8), enemy(8x8)]

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        logger.debug(f"Build Model with %d Res Blocks" % mc.res_layer_num)

        for _ in range(mc.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="connect4_model")

    def _build_residual_block(self, x):
        mc = self.config.model
        in_x = x
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", kernel_regularizer=l2(mc.l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"loading model from {weight_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {weight_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        #self.model.save(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

        '''Harvey Add'''

    def update_model(self, target):
        #copy_layers = len(m.layers)

        #print(copy_layers)
        copy_layers = target.get_num_res_layers() * 7 + 4
        #print(copy_layers)
        #copy_layers = int(copy_layers)
        model_layers = len(self.model.layers)
        target_layers = len(target.model.layers)
        # set the weights of the previous layers
        for i in range(copy_layers):
            #print(i)
            #print(target.model.layers[i])
            weight = target.model.layers[i].get_weights()
            self.model.layers[i].set_weights(weight)

        for i in range(5 * 2 + 1):
            #print(target_layers-i-1)
            #print(model_layers-i-1)
            weight = target.model.layers[target_layers-i-1].get_weights()
            self.model.layers[model_layers-i-1].set_weights(weight)

    def get_num_res_layers(self):
        print('model layers %f' % len(self.model.layers))
        return int((len(self.model.layers) - 4 - 5 * 2 - 1) / 7)

def objective_function_for_policy(y_true, y_pred):
    # can use categorical_crossentropy??
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)

def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


