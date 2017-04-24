#!/usr/bin/python

import os
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave, imread, imresize
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import time
import numpy as np

root_dir = os.path.abspath('.')
base_path = os.path.join(root_dir,'dog.jpg')
ref_path = os.path.join(root_dir,'style2.jpg')

EPOCHS = 5

img_nrows = 300
img_ncols = 300

style_wt = 0.5
content_wt = 0.025
total_var_wt = 0.5

def preprocess_img(path):
    print "preprocess: " + path
    i = load_img(path,target_size=(img_nrows,img_ncols))
    i = img_to_array(i)
    i = np.expand_dims(i,axis=0)
    i = vgg16.preprocess_input(i)
    return i

def deprocess_img(i):
    i = i.reshape((3,img_nrows,img_ncols))
    i = i.transpose((1,2,0))
    i[:,:,0] += 103.939
    i[:,:,1] += 116.779
    i[:,:,2] += 123.68
    i = i[:,:,::-1]
    i = np.clip(i,0,255).astype('uint8')
    return i

os.system("clear")

b_img = K.variable(preprocess_img(base_path))
r_img = K.variable(preprocess_img(ref_path))
final_img = K.placeholder((1,3,img_nrows,img_ncols))
input_tensor = K.concatenate([b_img, r_img, final_img],axis=0)

model = vgg16.VGG16(input_tensor=input_tensor,weights='imagenet',include_top=False)
#model.summary()
output_dict = dict([(layer.name,layer.output) for layer in model.layers])

def content_loss(base,final):
    return K.sum(K.square(final-base))		# square error

def gram_matrix(x):
    features = K.batch_flatten(x)		#flatten -> converts multidimension matrix to single dimensional array
    gram = K.dot(features,K.transpose(features))
    return gram

def style_loss(style,final):
    s = gram_matrix(style)
    f = gram_matrix(final)
    channels = 3				# RGB image
    size = img_nrows * img_ncols
    return K.sum(K.square(s-f))/(4.0*(channels**2)*(size**2)) # ** is 'raise to' operator in python.

def total_var_loss(x):
    a = K.square(x[:,:,:img_nrows-1,:img_ncols-1] - x[:,:,1:,:img_ncols-1])
    b = K.square(x[:,:,:img_nrows-1,:img_ncols-1] - x[:,:,:img_nrows-1,1:])
    return K.sum(K.pow(a+b,1.25))

loss = K.variable(0.0)
layer_features = output_dict['block4_conv2']
b_img_features = layer_features[0,:,:,:]
final_features = layer_features[2,:,:,:]
loss += content_wt * content_loss(b_img_features,final_features)

feature_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']

for layer in feature_layers:
    layer_features = output_dict[layer]
    style_features = layer_features[1,:,:,:]
    final_features = layer_features[2,:,:,:]
    sl = style_loss(style_features,final_features)
    loss += (style_wt / len(feature_layers))*sl

loss += total_var_wt*total_var_loss(final_img)

grads = K.gradients(loss,final_img)
outputs = [loss]
outputs.append(grads)
f_outputs = K.function([final_img],outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1,3,img_nrows,img_ncols))
    outs = f_outputs([x])
    loss_value = outs[0]
    if(len(outs[1:])==1):
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value,grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self,x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self,x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
x = preprocess_img(base_path)
z = load_img(base_path,target_size=(img_nrows,img_ncols))
imsave('iteration_original.png',z)
print "Saving original image.."

for i in range(EPOCHS):
    print "Start iteration: %d" % i
    start_time = time.time()
    x,min_val,info = fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime=evaluator.grads,maxfun=20)
    print "Loss: %s" % str(min_val)
    img = deprocess_img(x.copy())
    fname = "iteration_%d.png" % i
    imsave(fname,img)
    print "Time taken: %s" % str(time.time()-start_time)
