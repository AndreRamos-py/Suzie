from operator import truediv
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Embedding
from keras.utils.np_utils import to_categorical


data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}|{}'.format(command['entity'], command['action']))

#-----Process Text: Words, Characters, Bytes, SubWords


max_seq = max([len(bytes(x.encode('utf-8'))) for x in inputs])

print('Maior seq:', max_seq)

#-----Create the Dataset One-Hot (number of examples, length of sequence, number of characters)


#-----Input Data One-Hot Encoding-----#

input_data = np.zeros((len(inputs), max_seq, 256), dtype='float32')

for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

#-----Input Data Sparse-----#

'''
input_data = np.zeros((len(inputs), max_seq), dtype='int32')

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k] = chr2idx[ch]
'''

#-----Output Data-----#

labels = set(outputs)

fwrite = open('labels.txt', 'w', encoding='utf-8')

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label
    fwrite.write(label + '\n')
fwrite.close()

output_data = []

for output in outputs:
    output_data.append(label2idx[output])


output_data = to_categorical(output_data, len(output_data))

print(output_data[0])

model = Sequential()
model.add(LSTM(128))
model.add(Dense(len(output_data), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(input_data, output_data, epochs=128)

#-----Save Model-----#

model.save('model.h5')

#-----Sort text into a unit-----#

def classify(text):
    #Create an Input Array
    x = np.zeros((1, 48, 256), dtype='float32')
    
    #Fill array with data from text
    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0, k, int(ch)] = 1.0

    #Make the Forecast
    out = model.predict(x)
    idx = out.argmax()
    print(idx2label[idx])

'''
print(inputs)
print(outputs)
'''