import numpy as np
import tensorflow as tf
import keras
from keras.src.layers import TextVectorization 
# from keras.models import Model ,load_model 
# from keras.layers import Dense, Input, Dropout, Flatten,LSTM, Activation,Embedding,Bidirectional,Conv1D,MaxPooling1D
# from keras.preprocessing import sequence
# from keras.initializers import glorot_uniform
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.src.utils import to_categorical

from OptimiserSSA import NO_ITERATIONS, OptimizerSSA, build_model
#from sparrow import NO_ITERATIONS, OptimizerSSA, build_model

df1 = pandas.read_csv('human_chatgpt_genarated_dataset.csv')
X = df1.iloc[:, 0]
Y  = df1.iloc[:, 1]
print ('DataSet Size:'+str(len(X))+ ' Records')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)
#output sequence length max_length = 600 
max_length = 600 

text_vectorization = TextVectorization(ngrams=2, max_tokens=20000, 
                                       output_mode="int", 
                                       output_sequence_length=max_length,
                                       )
text_vectorization.adapt(X)
print ( 'vocabulary size:' + str(len(text_vectorization.get_vocabulary())))

encode_X_train = text_vectorization(X_train)
encode_X_test = text_vectorization(X_test)

#this encoding is must for loss  = 'categorical_crossentropy'
encode_y_train = to_categorical(Y_train)
encode_y_test = to_categorical(Y_test)

#model = get_model(max_tokens = len(text_vectorization.get_vocabulary()))
model = build_model(max_tokens = 600)
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("NLP_CNN02.keras",save_best_only=True)]
model.fit(encode_X_train,encode_y_train, epochs=10, callbacks=callbacks)
model.save_weights('NLP_CNN02.weights.h5')

trinscore = model.evaluate(encode_X_train,encode_y_train,batch_size=32, verbose=0)
testscore = model.evaluate(encode_X_test,encode_y_test,batch_size=32, verbose=0)
print("rmsprop -- train: {:.4f}  test: {:.4f}".format(trinscore, testscore))

model_p =  build_model(max_tokens = 600)

# Instantiate optimizer with model, loss function, and hyperparameters
ssa = OptimizerSSA(model=model_p, max_tokens = 600)  

# Train model on provided data
ssa.fit(encode_X_train, encode_y_train, steps=NO_ITERATIONS, batch_size=32)

# Get a copy of the model with the globally best weights
model_p = ssa.get_best_model()
model_p.save_weights('NLP_CNN02.SSA.weights.h5')
p_train_score = model_p.evaluate(encode_X_train, encode_y_train, batch_size=32, verbose=0)
p_test_score = model_p.evaluate(encode_X_test, encode_y_test, batch_size=32, verbose=0)
print("SSA -- train: {:.4f}  test: {:.4f}".format(p_train_score, p_test_score))
