import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train,X_test = X_train/255.0,X_test/255.0
print("X_train.shape:",X_train.shape)

