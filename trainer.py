from ast import Return
import random
import tensorflow as tf
import math
import numpy as np

# for interactive autocompliation
keras = tf.keras
layers = tf.keras.layers


tanh_model = keras.Sequential(layers=[
    layers.Dense(20, activation=keras.activations.relu),
    layers.Dense(50, activation=keras.activations.relu, use_bias=False),
    layers.Dense(30, activation=keras.activations.relu),
    layers.Dense(10, activation=keras.activations.relu),
    layers.Dense(1)
], name='tanh')

tanh_model.build((None, 1))
print (tanh_model.summary())

atanh_model = keras.Sequential(layers=[
    layers.Dense(20, activation=keras.activations.relu),
    layers.Dense(50, activation=keras.activations.relu, use_bias=False),
    layers.Dense(30, activation=keras.activations.relu),
    layers.Dense(10, activation=keras.activations.relu),
    layers.Dense(1)
], name='atanh')


atanh_model.build((None, 1))
print (atanh_model.summary())




loss_object = tf.keras.losses.mean_squared_error
loss_object_trainer = tf.keras.losses.mean_squared_error


optimizer = tf.keras.optimizers.Adam()
optimizer_trainer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_accuracy')

train_loss_trainer = tf.keras.metrics.Mean(name='trainer_loss')
train_accuracy_trainer = tf.keras.metrics.MeanAbsoluteError(name='trainer_accuracy')



@tf.function
def tanh_step(x):
    return    tanh_model(x)

@tf.function
def tanh_train_step(x):
    with tf.GradientTape() as tape:
        atanh_model.trainable = False
        y = tanh_model(x)
        predictions = atanh_model(y)
        loss = loss_object(x, predictions)

    gradients = tape.gradient(loss, tanh_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tanh_model.trainable_variables))

    train_loss(loss)
    train_accuracy(x, predictions)

    return y

   


@tf.function
def atanh_train_step(x, y):
    with tf.GradientTape() as tape:
        atanh_model.trainable = True
       
        predictions = atanh_model(x)
        loss = loss_object_trainer(x, predictions)

    gradients = tape.gradient(loss, atanh_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, atanh_model.trainable_variables))

    train_loss_trainer(loss)
    train_accuracy_trainer(y, predictions)

   


    

EPOCHS = 50

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    train_loss_trainer.reset_states()
    train_accuracy_trainer.reset_states()


    xs = []
    ys = []

    # print('Runing tanh training')
    for _ in range(10):
        x = np.random.random(size=(1,1))
        y = tanh_train_step(x)
        xs.append(x)
        ys.append(y)

    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    
    # print('Runing atanh training')
    atanh_train_step(ys, np.tanh(ys))

    # print('Updating embeding', tanh_model.layers[1].name)
    tanh_model.layers[1].set_weights(atanh_model.layers[1].get_weights())



    current_decayed_lr = optimizer._decayed_lr(tf.float32).numpy()
    current_decayed_lr_trainer = optimizer_trainer._decayed_lr(tf.float32).numpy()
    print(
        f'Epoch {epoch + 1}, '
        f'LR: {current_decayed_lr:7.3e}, '
        f'Loss: {train_loss.result():7.3e}, '
        f'Abs Erorr : {train_accuracy.result() * 100: 7.3}, '
        f'||||| Trainer LR: {current_decayed_lr:7.3e}, '
        f'Loss: {train_loss_trainer.result():7.3e}, '
        f'Abs Error : {train_accuracy_trainer.result() * 100: 7.3}'
    )
