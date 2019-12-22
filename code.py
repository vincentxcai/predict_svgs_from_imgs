# Deep Learning Shared Task
# 
# Xiaohong Cai 
# Anr: u955707
# Snr: 2023958 
# CodaLab name: jukebox

import numpy as np
import os
from natsort import natsorted
from matplotlib import image
import cv2
from keras import models, layers, regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# 1. Importing images

def load_imgs(path):
    '''load pngs into a numpy array'''
    imgs = []
    for filename in natsorted(os.listdir(path)): 
        img = image.imread(path + filename)
        img = cv2.resize(img,(32,32)) 
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

train_imgs = load_imgs('train/png/')
test_imgs = load_imgs('test/png/')

print('train_imgs shape:', train_imgs.shape)
print('test_imgs shape:', test_imgs.shape)

#----------------------------------------------------------------------------#

# 2. Importing and processing svg codes

# Read svg codes as strings
def load_svgs(path): 
    '''load svgs as strings'''
    svgs = []
    for filename in natsorted(os.listdir(path)):
        with open(path + filename) as file:
            text = file.read()
            svgs.append(text)
    return svgs

train_svgs = load_svgs('train/svg/') 
print('Number of svgs:', len(train_svgs))
print('First svg:', train_svgs[0])

# Identical part of a svg
head = '''<?xml version="1.0" encoding="utf-8" ?>
<svg baseProfile="full" height="64" version="1.1" width="64" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">'''

# Extract variable part of each svg
train_texts = []
for train_svg in train_svgs:
    train_texts.append(train_svg[len(head):])

print('Number of strings:', len(train_texts))
print('First string:', train_texts[0])

# Convert strings to sequences of indices
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(train_texts)
vocab_size = len(tokenizer.word_index) + 1
print('vocab size:', vocab_size)

index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])

train_sequences = tokenizer.texts_to_sequences(train_texts)
print('Number of sequences',len(train_sequences))
print('First sequence:', train_sequences[0])

# Find the longest sequence
maxlen = len(max(train_sequences, key=len))
print('Max length of sequences:', maxlen)

#----------------------------------------------------------------------------#

# 3. Preprocessing inputs and outputs for training
def preprocess_data(images, sequences): 
    '''preparing images,sequences and indices for training'''
    input_imgs = [] 
    input_seqs = [] 
    outputs = []
    for img, seq in zip(images, sequences):
        for i in range(1, len(seq)):
            input_seq, output = seq[:i], seq[i]
            input_seq = pad_sequences([input_seq],maxlen=maxlen).flatten()
            output = to_categorical(output,num_classes = vocab_size) 
            input_imgs.append(img)
            input_seqs.append(input_seq)
            outputs.append(output)
    input_imgs, input_seqs, outputs = np.array(input_imgs), np.array(input_seqs), np.array(outputs)
    print(" {} {} {}".format(input_imgs.shape,input_seqs.shape,outputs.shape))
    return input_imgs, input_seqs, outputs

input_imgs, input_seqs, outputs = preprocess_data(train_imgs[:46000], train_sequences[:46000])
input_imgs_val, input_seqs_val, outputs_val = preprocess_data(train_imgs[46000:], train_sequences[46000:])

#----------------------------------------------------------------------------#

# 4. Building the model 

# Encode images
input_img = layers.Input(shape=train_imgs.shape[1:])
img = layers.Conv2D(16, 3, padding='same', activation='relu', strides=(2,2))(input_img)
img = layers.Conv2D(32, 3, padding='same', activation='relu', strides=(2,2))(img)
img = layers.Flatten()(img)
img = layers.Dense(128, activation='relu')(img)

# Encode sequences
input_seq = layers.Input(shape=(maxlen,))
seq = layers.Embedding(vocab_size, 8, mask_zero=True)(input_seq)
seq = layers.LSTM(128, return_sequences=True)(seq)
seq = layers.Dropout(0.05)(seq)
seq = layers.LSTM(128)(seq)

# Decoder
decoder = layers.add([img, seq])
decoder = layers.Dense(128,activation='relu')(decoder)
output = layers.Dense(vocab_size, activation='softmax')(decoder)

# Complete model
model = models.Model([input_img, input_seq], output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())

#----------------------------------------------------------------------------#

# 5. Training the model 
# Save the model after each epoch for safety
for i in range(1,8):
  print('epoch {}'.format(i))
  model.fit([input_imgs, input_seqs], outputs, 
                    epochs=1,
                    batch_size=128,
                    validation_data=([input_imgs_val, input_seqs_val], outputs_val))  
  model.save('my_model_epoch{}.h5'.format(i))

#----------------------------------------------------------------------------#
  
# 6. Making predictions

def predict_svg(image):
    '''Predict svg code from an image'''
    text = '<defs'
    for i in range(maxlen):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq],maxlen)
        pred = model.predict([[image][0:1], seq], verbose=0)
        pred = np.argmax(pred)
        word = index_word[pred]
        text += " " + word
        if word == '/></svg>':
            break
    return head + text    

# Creating svg files
for i, img in enumerate(test_imgs):
    with open(os.path.join('test/svg/',str(i+48000)+'.svg'), "w") as f:
        f.write(predict_svg(img))
