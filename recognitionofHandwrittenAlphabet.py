'''
A Multilayer Perceptron implementation example using TensorFlow library.
Project

Author: GAO Feng

'''

from __future__ import print_function

import sys

reload(sys)

sys.setdefaultencoding('utf-8')

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import random

from sklearn.feature_extraction import DictVectorizer



train = pd.read_csv('data/letter_recognition_training_data_set.csv')

test = pd.read_csv('data/letter_recognition_testing_data_set.csv')

feature_to_input=['x-box', 'y-box' , 'width' , 'high' , 'onpix' , 'x-bar' , 'y-bar' , 'x2bar' , 'y2bar' , 'xybar' , 'x2ybr', 'xy2br' , 'x-ege' , 'xegvy' , 'y-ege' , 'yegvx' ]

feature_to_predict=['label']

# Parameters

learning_rate = 0.01

training_epochs = 10

batch_size = 80

display_step = 1



# Network Parameters

n_hidden_1 =300 # 1st layer number of features

n_hidden_2 =300



n_input = len(feature_to_input) # data input 16

n_classes = 26 #total classes 26



split_size=16000



train_input= train[feature_to_input]

train_lable= train[feature_to_predict]

test_input=test[feature_to_input]



train_x, val_x = train_input[0:split_size], train_input[split_size:]

train_y, val_y = train_lable[0:split_size], train_lable[split_size:]



train_x=[list(x) for x in train_x.values]

val_x=[list(x) for x in val_x.values]

train_y=[list(x) for x in train_y.values]

val_y=[list(x) for x in val_y.values]

test_x=[list(x) for x in test_input.values]



train_x=np.array(train_x)

val_x==np.array(val_x)

train_y=np.array(train_y)

val_y=np.array(val_y)

test_x=np.array(test_x)



def one_hot(train_lable,split):

   train_lable=train_lable.reshape((split,))

   X = pd.DataFrame({'Lable':train_lable})

   v = DictVectorizer()

   X_qual = v.fit_transform(X[['Lable']].to_dict('records'))

   v.vocabulary_

   {'Lable=A': 0,'Lable=B': 1, 'Lable=C': 2 ,'Lable=D': 3,'Lable=E': 4,'Lable=F': 5,'Lable=G': 6,

    'Lable=H': 7, 'Lable=I': 8,'Lable=J': 9,'Lable=K': 10,'Lable=L': 11,'Lable=M': 12,'Lable=N': 13,

    'Lable=O': 14,'Lable=P':15,'Lable=Q': 16,'Lable=R': 17,'Lable=S': 18,'Lable=T': 19,'Lable=U': 20,'Lable=V': 21,'Lable=W': 22,'Lable=X': 23,'Lable=Y': 24,'Lable=Z': 25}

   return X_qual.toarray()

   

train_y=one_hot(train_y,split_size)

val_y=one_hot(val_y,20000-split_size)



with tf.name_scope('inputs'):

    x = tf.placeholder("float", [None, n_input], name='x_in')

    y = tf.placeholder("float", [None, n_classes], name='y_in')







# Create model

def multilayer_perceptron(x, weights, biases):

    # Hidden layer with RELU activation

    with tf.name_scope('hidden_layer1'):

        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

        layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation

    with tf.name_scope('hidden_layer2'):

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

        layer_2 = tf.nn.relu(layer_2)

    

    # Output layer with linear activation

    with tf.name_scope('prediction'):

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

    

def next_batch(batch_size,i):  

    arr=T[i*batch_size :(i+1)*batch_size]

    batch_x= train_x[arr,:]

    batch_y= train_y[arr,:]

    return batch_x, batch_y  





# Store layers weight & bias

with tf.name_scope('weights'):

    weights = {

        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='weight1'),

        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='weight2'),

        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='weight_ouput')

    }

with tf.name_scope('biases'):

    biases = {

        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='biase1'),

        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='biase2'),

        'out': tf.Variable(tf.random_normal([n_classes]), name='biase_output')

    }



   

# Construct model

pred = multilayer_perceptron(x, weights, biases)



# Define loss and optimizer

with tf.name_scope('cross_entrop'):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

with tf.name_scope('train'):	

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Initializing the variables

init = tf.global_variables_initializer()

T=random.sample(range(0,split_size),split_size) 

validation_acc=np.zeros(training_epochs+1)

training_cost=np.zeros(training_epochs+1)



# Launch the graph

with tf.Session() as sess:

    sess.run(init)

    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        batch_num = int(split_size/batch_size)

        # Loop over all batches

        for i in range(0,batch_num):

#           

          batch_x, batch_y = next_batch(batch_size,i)

            # Run optimization op (backprop) and cost op (to get loss value)\\

          _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

              # Compute average loss

          avg_cost += c / batch_num

        # Display logs per epoch step

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1), "cost=", \

                "{:.9f}".format(avg_cost))

#    print("Optimization Finished!")

    # Test model

        training_cost[epoch+1]=avg_cost

        with tf.name_scope('accuracy'):

            with tf.name_scope('correct_prediction'):

                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

  #  Calculate accuracy

            with tf.name_scope('accuracy'):

                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                acc=accuracy.eval({x: val_x, y: val_y})        

                print("Accuracy:", acc)

                validation_acc[epoch+1]=acc

    #test/testing_data_set.csv

    classification = sess.run(tf.argmax(pred, 1), feed_dict={x:test_x})

    #save neural network graph

    writer = tf.summary.FileWriter("/home/bigdata/project/",sess.graph)

    

    plt.figure(1)

    plt.xlabel('Epoch')

    plt.ylabel('Validation Accuracy')

    plt.title('Validation accuracy with different epoch')

    plt.margins(0.08)

    plt.subplots_adjust(bottom=0.15)

    plt.plot(validation_acc,'r')

    plt.show()

    plt.figure(2)

    plt.xlabel('Epoch')

    plt.ylabel('Training average cost')

    plt.title('Training average cost with different epoch')

    plt.margins(0.08)

    plt.subplots_adjust(bottom=0.15)

    plt.plot(training_cost,'r')

    plt.show()

    

    sum=a=b=c=d=e=f=g=h=i=j=k=l=m=n=o=p=q=r=s=t=u=v=w=x=y=z=0

    for num in classification:

     if num==0:

      a=a+1

     elif num==1:

      b=b+1

     elif num==2:

      c=c+1

     elif num==3:

      d=d+1

     elif num==4:

      e=e+1

     elif num==5:

      f=f+1

     elif num==6:

      g=g+1

     elif num==7:

      h=h+1

     elif num==8:

      i=i+1

     elif num==9:

      j=j+1

     elif num==10:

      k=k+1

     elif num==11:

      l=l+1

     elif num==12:

      m=m+1

     elif num==13:

      n=n+1

     elif num==14:

      o=o+1

     elif num==15:

      p=p+1

     elif num==16:

      q=q+1

     elif num==17:

      r=r+1    

     elif num==18:

      s=s+1  

     elif num==19:

      t=t+1  

     elif num==20:

      u=u+1  

     elif num==21:

      v=v+1  

     elif num==22:

      w=w+1  

     elif num==23:

      x=x+1  

     elif num==24:

      y=y+1  

     elif num==25:

      z=z+1

     else:

      print(0)

    sum=a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z

#    print(a,b,c,d,e,f,g,h,i,j,sum)

plt.figure(3)

lables_numbers=
[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]

print("The numbers of relative alphabet: ", lables_numbers)

lables = ['A', 'B', 'C', 'D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

x = range(len(lables))

plt.plot(x, lables_numbers, 'ro-')

plt.xticks(x, lables, rotation=10)

plt.margins(0.08)

plt.title('Classification of testing dataset')

plt.subplots_adjust(bottom=0.15)

plt.show()