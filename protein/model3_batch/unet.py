__author__ = 'Bingqing Wei'
from keras.layers import *

'''
Code by https://github.com/zhixuhao/unet
'''

def unet(inputs):
    conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)

    conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv1D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(drop5))
    merge6 = concatenate([drop4, up6], axis=-1)
    conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv1D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling1D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv1D(1, 1, activation='sigmoid')(conv9)

    x = TimeDistributed(Dense(3))(conv10)
    return x