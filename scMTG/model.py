import os
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Lambda, Layer

# MeanAct = lambda x: tf.clip_by_value(K.exp(x), 0, 1e4)
MeanAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 0, 1e4)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


class BaseFullyConnectedNet(tf.keras.Model):
    """ Generator network.
    """
    def __init__(self, input_dim, z_dim, output_dim, model_name, nb_units=[256], concat_every_fcl=False, batchnorm=False, dropout=True, last_relu=False):  
        super(BaseFullyConnectedNet, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.last_relu = last_relu
        self.all_layers = []
        """ Builds the FC stacks. """
        for i in range(len(nb_units)):
            fc_layer = tf.keras.layers.Dense(
                units = self.nb_units[i],
                activation = None,
#                 kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )  
            norm_layer = tf.keras.layers.BatchNormalization()
            dropout_layer = tf.keras.layers.Dropout(0.1)
            act_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
#             act_layer = tf.keras.layers.ReLU()
            self.all_layers.append([fc_layer, norm_layer, dropout_layer, act_layer])
        fc_layer = tf.keras.layers.Dense(
            units = self.output_dim,
            activation = None,
#             kernel_regularizer = tf.keras.regularizers.L2(2.5e-5),
#             activity_regularizer = tf.keras.regularizers.L1(2.5e-3)
        )
        self.all_layers.append([fc_layer, None, None, None])
        
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        y = inputs[:,self.z_dim:]
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer, dropout_layer, act_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                if self.dropout:
                    x = dropout_layer(x)
                x = act_layer(x)
                if self.concat_every_fcl:
                    x = tf.keras.layers.concatenate([x,y],axis=1)
        fc_layer, _, _, _ = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
            # No activation func at last layer
            if self.last_relu:
#                 output = tf.keras.layers.ReLU()(output)
                output = tf.keras.activations.tanh(output)
        return output
    
class Decoder2(tf.keras.Model):
    """ Decoder network.
    """
    def __init__(self, input_dim, z_dim, output_dim, model_name, nb_units=[256], batchnorm=False, dropout=True, last_relu=False):  
        super(Decoder2, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.last_relu = last_relu
        self.all_layers = []
        """ Builds the FC stacks. """
        for i in range(len(nb_units)):
            fc_layer = tf.keras.layers.Dense(
                units = self.nb_units[i],
                activation = None,
#                 kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )  
            norm_layer = tf.keras.layers.BatchNormalization()
            dropout_layer = tf.keras.layers.Dropout(0.1)
            act_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
#             act_layer = tf.keras.layers.ReLU()
            self.all_layers.append([fc_layer, norm_layer, dropout_layer, act_layer])
            
        disp_layer = tf.keras.layers.Dense(
            units=self.output_dim, 
            activation=DispAct, 
#             kernel_initializer='glorot_uniform', 
            name='dispersion'
        )
        mean_layer = tf.keras.layers.Dense(
            units=self.output_dim, 
            activation=MeanAct, 
#             kernel_initializer='glorot_uniform', 
            name='mean'
        )
        self.all_layers.append([disp_layer, mean_layer])
        
        self.disp, self.mean = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer, dropout_layer, act_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                if self.dropout:
                    x = dropout_layer(x)
                x = act_layer(x)
        disp_layer, mean_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_output" % self.model_name):
            disp = disp_layer(x)
            mean = mean_layer(x)
            # No activation func at last layer
            if self.last_relu:
                mean = tf.keras.layers.ReLU()(mean)
        return disp, mean
    
class Generator(tf.keras.Model):
    """Generator network.
    """
    def __init__(self, input_dim, z_dim, output_dim, model_name, nb_units=[256], concat_every_fcl=False, batchnorm=False, dropout=True, last_relu=False):  
        super(Generator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.concat_every_fcl = concat_every_fcl
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.last_relu = last_relu
        self.scale_factor = tf.Variable(initial_value=3.0, trainable=True)
        self.all_layers = []
        
        """Builds the FC stacks."""
        for i in range(len(self.nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
#                 kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            dropout_layer = tf.keras.layers.Dropout(0.1)
            self.all_layers.append([fc_layer, norm_layer, dropout_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, z_dim + nb_classes]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        y = inputs[:,self.z_dim:]
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer, dropout_layer = layers
            with tf.name_scope("%s_g_layer_%d" % (self.model_name,i)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
#                 if i==len(self.nb_units):
#                     x = norm_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                if self.dropout:
                    x = dropout_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#                 x = tf.keras.layers.ReLU()(x)
                if self.concat_every_fcl:
                    x = tf.keras.layers.concatenate([x,y],axis=1)
        fc_layer, norm_layer, dropout_layer = self.all_layers[-1]
        with tf.name_scope("%s_g_layer_output"%self.model_name):
            output = fc_layer(x)
            # No activation func at last layer
            if self.last_relu:
#                 output = tf.keras.layers.ReLU()(output)
#                 output = tf.keras.activations.tanh(output)
                output = self.scale_factor * tf.keras.activations.tanh(output)
        return output
    
class Discriminator(tf.keras.Model):
    """Discriminator network.
    """
    def __init__(self, input_dim, z_dim, output_dim, model_name, nb_units=[256], batchnorm=False, dropout=True, last_relu=False):  
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.last_relu = last_relu
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
#                 kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            dropout_layer = tf.keras.layers.Dropout(0.1)
            self.all_layers.append([fc_layer, norm_layer, dropout_layer])
            
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
        fc_layer, norm_layer, dropout_layer = self.all_layers[0]
        with tf.name_scope("%s_d_layer_0" % self.model_name):
            x = fc_layer(inputs)
            if self.dropout:
                x = dropout_layer(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#             x = tf.keras.layers.ReLU()(x)
            
        for i, layers in enumerate(self.all_layers[1:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer, dropout_layer = layers
            with tf.name_scope("%s_d_layer_%d" % (self.model_name,i+1)):
                x = fc_layer(x)
#                 if i==len(self.nb_units):
#                     x = norm_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                if self.dropout:
                    x = dropout_layer(x)
#                 x = tf.keras.activations.tanh(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
#                 x = tf.keras.layers.ReLU()(x)
        fc_layer, norm_layer, dropout_layer = self.all_layers[-1]
        with tf.name_scope("%s_d_layer_output" % self.model_name):
            output = fc_layer(x)
            if self.last_relu:
                output = tf.keras.layers.ReLU()(output)
        return output