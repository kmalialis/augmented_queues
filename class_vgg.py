import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten


class VGG:

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            input_shape,
            num_classes,
            learning_rate,
            num_epochs,
            num_blocks=2,
            init_filters=64,
            num_fc_layers=2,
            activation='relu',
            output_activation='softmax',
            loss_function='categorical_crossentropy',
            batch_size=64,
            l2_reg=0.0,
            dropout=0.0,
            seed=0
    ):

        self.input_shape = input_shape

        self.num_blocks = num_blocks
        self.init_filters = init_filters
        self.num_fc_layers = num_fc_layers

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.activation = activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.seed = seed

        self.weight_init = he_normal(seed=self.seed)
        self.weight_init_out = glorot_uniform(seed=self.seed)

        # model
        self.model = self.create_vgg_model()

        self.model.summary()

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.loss_function
        )

    ##############
    # Prediction #
    ##############

    def predict(self, x):
        y_hat = self.model.predict(x=x, verbose=0)  # probability of each class
        y_hat_max = np.max(y_hat, axis=1)  # highest probability
        y_hat_argmax = np.argmax(y_hat, axis=1)  # predicted class

        return y_hat, y_hat_max, y_hat_argmax

    ############
    # Training #
    ############

    def train(self, x=None, y=None, validation_data=None, flag_shuffle=True, verbose=0):
        self.model.fit(
            x=x,
            y=y,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose  # 0: off, 1: full, 2: brief
        )

    ###########################################################################################
    #                                      Auxiliary                                          #
    ###########################################################################################

    #############
    # VGG Model #
    #############

    def create_vgg_model(self):
        # Input and output dims
        input_shape = self.input_shape
        num_classes = self.num_classes
        num_blocks = self.num_blocks
        filters = self.init_filters
        num_fc_layers = self.num_fc_layers
        dropout_rate = self.dropout

        # Input layer
        x_input = Input(shape=input_shape, name='input')
        x = None

        for block_id in range(num_blocks):

            if block_id == 0:
                x = x_input

            # Add two convolutional layers
            for i in range(2):
                x = Conv2D(filters=filters * (2**block_id),
                           kernel_size=(1, 3),  # 3,3
                           strides=(1, 1),
                           padding="same",
                           input_shape=input_shape,
                           activation=self.activation,
                           kernel_initializer=self.weight_init,
                           name=f'block{block_id+1}_conv{i+1}')(x)

            # Add a third convolutional layer from the third block and onwards
            if block_id >= 2:
                print("filters", filters * (2**block_id))
                x = Conv2D(filters=filters * (2**block_id),
                           kernel_size=(1, 3),  # 3,3
                           strides=(1, 1),
                           padding="same",
                           input_shape=input_shape,
                           activation=self.activation,
                           kernel_initializer=self.weight_init,
                           name=f'block{block_id+1}_conv3')(x)

            x = MaxPooling2D(pool_size=(1, 2),  # 2,2
                             strides=(1, 2),  # 2,2
                             name=f'block{block_id+1}_pool')(x)

        x = Flatten()(x)
        for i in range(num_fc_layers):
            x = Dense(units=filters * (2**(num_blocks-1)) * 8,
                      activation=self.activation,
                      kernel_initializer=self.weight_init,
                      name=f"fc{i+1}")(x)

        if dropout_rate > 0:
            x = Dropout(rate=dropout_rate)(x)

        # softmax classifier
        y_out = Dense(units=num_classes,
                      activation=self.output_activation,
                      kernel_initializer=self.weight_init_out,
                      name='predictions')(x)

        # Model
        return Model(inputs=x_input, outputs=y_out)
