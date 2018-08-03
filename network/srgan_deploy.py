import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Add
from keras.layers import BatchNormalization, Conv2D
from keras.layers import UpSampling2D
from keras.optimizers import Adam

class SRGAN():
    def __init__(self, height_lr=64, width_lr=64, channels=3, upscaling_factor=4, gen_lr=1e-4):
        """
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        :param int gan_lr: Learning rate of GAN
        """

        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr

        # High-resolution image dimensions
        if upscaling_factor % 2 != 0:
            raise ValueError('Upscaling factor must be a multiple of 2; i.e. 2, 4, 8, etc.')
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        optimizer_generator = Adam(gen_lr, 0.9)
        self.generator = self.build_generator(optimizer_generator)
        self.generator.load_weights('imagenet_generator.h5')


    def build_generator(self, optimizer, residual_blocks=16):
        """
        Build the generator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """

        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def deconv2d_block(input):
            x = UpSampling2D(size=2)(input)
            x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
            x = Activation('relu')(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same')(lr_input)
        x_start = Activation('relu')(x_start)

        # Residual blocks
        r = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            r = residual_block(r)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([x, x_start])

        # Upsampling (if 4; run twice, if 8; run thrice, etc.)
        for _ in range(int(np.log(self.upscaling_factor) / np.log(2))):
            x = deconv2d_block(x)

        # Generate high resolution output
        hr_output = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer
        )
        return model
