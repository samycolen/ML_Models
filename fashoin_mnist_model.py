#?  each picture is consist of certain numbers of pixels with values between O to 255
#?  Any image/picture is consist of rectangular grid of pixels 

#?  If we take an example like an image is of 28x28 grid size of pixels so this image is " a set of 784 values between O to 255 " and we can say that 
#?  these are our Input ( X )

#?  like we have 10 items in our dataset lets say these are Y (o/p)


import tensorflow as tf


data = tf.keras.datasets.fashion_mnist 

#! about fashion_mnist : 

#! training_images : is an Array which contains 60000 "28x28" arrays. 
#! training_labels : is an Array which contains 60000  values between (0-9)


#! test_images     : is an Array which contains 10000  "28X28" arrays  
#! test_labels     : is an Array which contains 10000  values between (0-9)


(training_images, training_labels), (test_images, test_labels) = data.load_data()   #! -------> loading the data

training_images = training_images / 255.0   #!........................>      Normalizing the data
test_images = test_images / 255.0           #!........................>      Normalizing the data




#! crating the model 
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),                  #!  input layer specification.
tf.keras.layers.Dense(128, activation=tf.nn.relu),              #!  middle layer of neurons which contains 128 neurons(hidden layer)
tf.keras.layers.Dense(10, activation=tf.nn.softmax)             #!  o/p layer of 10 neurons (because we have 10 classes) 
])


#! Flatten convert this 2D array into linear data array(1D)
#! relu is an activation function in middle which returns the value which is greater than 0
#! each nueron of the o/p layer is end up with the probablity that the "input pixels match that class" and we are picking highest of them for this we are using softmax activation function



#!  here we are defining the optimizer ,loss fn and also metrics which is showing the accuracy
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


#! here we training our model using the fit() method : fitting the training_images to the training_labels over five epoches 


model.fit(training_images, training_labels, epochs=5)


#! after training the model we have to evaluate the model by passing the test_images and checking the test_laberls

model.evaluate(training_images,training_labels)
model.evaluate(test_images,test_labels)


#! after training and evaluating the model we have to explore our model 

classification = model.predict(test_images)  #!  it is a set 

print(classification[9999])

print(test_labels[9999])  
     