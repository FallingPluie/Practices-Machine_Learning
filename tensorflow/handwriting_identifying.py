print("初始化……")
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 初始化数据
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # 压缩数据
is9 = [0,0,0,0,0,0,0,0,0,1]
for i in range(len(y_train)):
    y_train[i] = is9[y_train[i]]
for i in range(len(y_test)):
    y_test[i] = is9[y_test[i]]
print("初始化完成")

# 检查数据
'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)  # type: ignore
    plt.xlabel(y_train[i])
plt.show()'''

# 训练数据
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
print("训练完成, 评估准确度:")
model.evaluate(x_test, y_test)

# 测试结果可视化
print("预测结果:")
start = 100
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
plt.figure(figsize=(10,10))
for i in range(start,start+25):
    plt.subplot(5,5,i+1-start)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap=plt.cm.binary) # type: ignore
    plt.xlabel(np.argmax(predictions[i])) # type: ignore
plt.show()