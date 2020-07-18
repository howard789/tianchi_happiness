from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# 图像分类
model = keras.Sequential(
[
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

optimizer=tf.keras.optimizers.Adam(0.001)
loss=tf.keras.losses.sparse_categorical_crossentropy
metrics=tf.keras.metrics.categorical_accuracy
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 创建网络
inputs = tf.keras.Input(shape=(784,), name='img')
h1 = layers.Dense(32, activation='relu')(inputs)
h2 = layers.Dense(32, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist model')

# 展示
model.summary()
keras.utils.plot_model(model, 'mnist_model.png')
keras.utils.plot_model(model, 'model_info.png', show_shapes=True)

# 训练
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)


# 评价
test_scores = model.evaluate(x_test, y_test, verbose=0)


# 模型保持和序列化
model.save('model_save.h5')
del model
model = keras.models.load_model('model_save.h5')

keras.experimental.export_saved_model(model, 'saved_model')
new_model = keras.experimental.load_from_saved_model('saved_model')



