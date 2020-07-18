from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import numpy as np
import tensorflow as tf

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(predict-ys),
#                        reduction_indices=[1]))

def amend_adam_rate(model,rate):
    opt = optimizers.Adam(rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


# def get_model_1(num_classes, max_words_per_sentence, max_word_id_num):
#     # #-1,2:正确率17.21%
#     model = models.Sequential()
#     model.add(layers.Embedding(input_dim=max_word_id_num + 1, output_dim=256, input_length=max_words_per_sentence))
#     model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), merge_mode="concat"))
#     model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), merge_mode="concat"))
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dense(num_classes, activation='softmax'))
#     opt = optimizers.Adam(0.001)
#     model.compile(loss="sparse_categorical_crossentropy", optimizer=opt)
#     return model

# def get_model_2(dims,num_classes):
#     #-1,2:正确率14.17% 正确率16.70%
#     # opt = optimizers.Adam(0.01)  正确率6.70%
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims))
#     model.add(layers.Dense(units=dims / 2, activation='relu'))
#     model.add(layers.Dense(units=dims / 4, activation='relu'))
#     model.add(layers.Dense(units=dims / 8, activation='relu'))
#     model.add(layers.Dense(units=dims / 10, activation='relu'))
#     model.add(layers.Dense(units=dims / 20, activation='relu'))
#     model.add(layers.Dense(units=num_classes))
#     # sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model



# def get_model_3(dims,num_classes):
#     # opt = optimizers.Adam(0.01)  正确率58.65%
#
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims / 2))
#     model.add(layers.Dense(units=dims / 4))
#     model.add(layers.Dense(units=dims / 8))
#     model.add(layers.Dense(units=dims / 10))
#     model.add(layers.Dense(units=dims / 20))
#     model.add(layers.Dense(units=num_classes))
#     # sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model

# def get_model_4(dims,num_classes):
#
#
#     # 32 -1  缩放 正确率13.88%
#
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims))
#     model.add(layers.Dense(units=160))
#     model.add(layers.Dense(units=80))
#     model.add(layers.Dense(units=40))
#     model.add(layers.Dense(units=20))
#     model.add(layers.Dense(units=10))
#     model.add(layers.Dense(units=num_classes))
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model


# def get_model_5(dims,num_classes):
#     #-1,2 :正确率59.64% 14.21%
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims))
#     model.add(layers.Dense(units=160))
#     model.add(layers.Dense(units=160, activation='relu'))
#     model.add(layers.Dense(units=80))
#     model.add(layers.Dense(units=40))
#     model.add(layers.Dense(units=40, activation='relu'))
#     model.add(layers.Dense(units=20))
#     model.add(layers.Dense(units=10))
#     model.add(layers.Dense(units=10, activation='relu'))
#     model.add(layers.Dense(units=num_classes))
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model



# def get_model_6(dims,num_classes):
#     #-1,2 :正确率37.42%
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims))
#     model.add(layers.Dense(units=160))
#     model.add(layers.Dense(units=80))
#     model.add(layers.Dense(units=40))
#     model.add(layers.Dense(units=20))
#     model.add(layers.Dense(units=10))
#     model.add(layers.Dense(units=num_classes,activation='softmax'))
#     opt = optimizers.Adam(0.01)
#     #
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['mse'])
#
#     # model.compile(loss='mse', optimizer='rmsprop')
#     return model

# def get_model_7(dims,num_classes):
#     #-1,2 :正确率1.25%
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims))
#     model.add(layers.Dense(units=160))
#     model.add(layers.Dense(units=80))
#     model.add(layers.Dense(units=40))
#     model.add(layers.Dense(units=20))
#     model.add(layers.Dense(units=10))
#     model.add(layers.Dense(units=num_classes))
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='mse',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#     return model


def get_model_8(dims,num_classes):


    # 32 -1  缩放 正确率60.98%
    model = models.Sequential()
    model.add(layers.Dense(units=dims,activation='sigmoid'))
    model.add(layers.Dense(units=160,activation='sigmoid'))
    model.add(layers.Dense(units=80,activation='sigmoid'))
    model.add(layers.Dense(units=40,activation='sigmoid'))
    model.add(layers.Dense(units=20,activation='sigmoid'))
    model.add(layers.Dense(units=10,activation='sigmoid'))
    model.add(layers.Dense(units=num_classes,activation='sigmoid'))
    opt = optimizers.Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model



def get_model_10(dims,num_classes):

    # 32 -1  缩放 正确率60.98%
    model = models.Sequential()
    model.add(layers.Dense(units=dims,activation='tanh'))
    model.add(layers.Dense(units=dims/2,activation='tanh'))
    model.add(layers.Dense(units=dims/4,activation='tanh'))
    model.add(layers.Dense(units=dims/8,activation='tanh'))
    model.add(layers.Dense(units=dims/16,activation='tanh'))
    model.add(layers.Dense(units=num_classes))
    opt = optimizers.Adam(0.01)
    # loss=tf.keras.losses.sparse_categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def get_model_11(dims, num_classes):

    # 1 -1 2 正确率61.30%
    # 32 -1 2 正确率61.30% 缩放后minus
    model = models.Sequential()
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=num_classes))
    opt = optimizers.Adam(0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


# def get_model_12(dims, num_classes):
#     # batch32 正确率59.82% [(0, 0.5982478097622027), (1, 0.5982478097622027), (2, 0.5982478097622027), (3, 0.5982478097622027)]
#     # batch1 [(0, 0.6052631578947368), (1, 0.6052631578947368), (2, 0.6052631578947368), (3, 0.6052631578947368)]
#     model = models.Sequential()
#     model.add(layers.Dense(units=dims, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/2, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/4, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/6, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/8, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/10, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=dims/20, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(units=num_classes, activation='softmax'))
#     opt = optimizers.Adam(0.01)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=opt,
#                   metrics=['accuracy'])
#
#     return model



def get_model_13(dims, num_classes):
    # 缩放后
    # batch_num=32  正确率60.85%
    # batch_num=32  正确率62.10%
    # batch_num=1 正确率61.30%
    model = models.Sequential()
    model.add(layers.Dense(units=dims/2, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=dims/6, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=dims/20, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    opt = optimizers.Adam(0.01)

    # loss_str='sparse_categorical_crossentropy'
    loss_str='categorical_crossentropy'

    model.compile(loss=loss_str,
                  optimizer=opt,
                  metrics=['accuracy'])





    return model




