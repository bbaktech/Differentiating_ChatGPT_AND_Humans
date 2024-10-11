import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import keras
from keras.src.layers import TextVectorization
from keras.src.models import Model
from keras.src.layers import Dense, Input, Dropout, LSTM, Activation,Embedding,Bidirectional

import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix

from OptimiserSSA import OptimizerSSA
LOSS = 'categorical_crossentropy' # Loss function
OPTIMISER = "rmsprop"

def get_model(max_tokens=23, hidden_dim=64):
    inputs = Input(shape=(None,), dtype="int64")
    embedded = Embedding( input_dim=max_tokens, output_dim=256, mask_zero=True) (inputs)
    x = Bidirectional(LSTM(32))(embedded)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="softmax")(x) # binary activation output(sigmoid:1,softmax":n))
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy")
    return model

df1 = pandas.read_csv('human_chatgpt_genarated_dataset.csv')
X = df1.iloc[:, 0]
Y  = df1.iloc[:, 1]

print ('DataSet Size:'+str(len(X))+ ' Records')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

max_length = 600 
max_tokens = 10000

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
    )

text_vectorization.adapt(X)
print ( 'vocabulary size:' + str(len(text_vectorization.get_vocabulary())))

LSTM_X_train = text_vectorization(X_train)
LSTM_X_test = text_vectorization(X_test)

#this encoding is must for loss  = 'categorical_crossentropy'
encode_y_train = to_categorical(Y_train)
encode_y_test = to_categorical(Y_test)

callbacks = [ keras.callbacks.ModelCheckpoint("LSTRNNMDL.keras",save_best_only=True)]
model = get_model(max_tokens = len(text_vectorization.get_vocabulary()))
model.summary()
#Histry = model.fit ( LSTM_X_train, encode_y_train,epochs=1)
Histry = model.fit ( LSTM_X_train, encode_y_train,epochs=5,verbose=0)
# print(Histry.history.keys())
# train_loss = Histry.history['loss']
score1 = model.evaluate(LSTM_X_train, encode_y_train)
score2 = model.evaluate(LSTM_X_test, encode_y_test)
#print("rmsprop -- train_loss: {:.4f}  test: {:.4f}".format(score1, score2))
print("rmsprop --> {:.4f}".format( score2))
model.save_weights('NLP_LSTRNNMDL.weights.h5')

model_p =  get_model(max_tokens = len(text_vectorization.get_vocabulary()))

# Instantiate optimizer with model, loss function, and hyperparameters
ssa = OptimizerSSA(model=model_p, loss= LOSS,optimiser = OPTIMISER )  

# Train model on provided data
#ssa.fit(LSTM_X_train, encode_y_train, batch_size=32)
ssa.fit(LSTM_X_train, encode_y_train, batch_size=32)

# Get a copy of the model with the globally best weights
model_p = ssa.get_best_model()
model_p.save_weights('NLP_LSTRNNMDL.SSA.weights.h5')
p_train_score = model_p.evaluate(LSTM_X_train, encode_y_train, batch_size=32, verbose=0)
p_test_score = model_p.evaluate(LSTM_X_test, encode_y_test, batch_size=32, verbose=0)
#print("SSA -- train: {:.4f}  test: {:.4f}".format(p_train_score, p_test_score))
print("SSA --> {:.4f}".format( p_test_score))
#code tyo genarate confusion_matrix
y_pred = []
pr = model_p.predict(LSTM_X_test)
for pi in pr:
    p = np.argmax(pi)
    y_pred.append(p)

print(X_test)
print(y_pred)

cm = confusion_matrix(Y_test, y_pred)
print('confusion_matrix')
print(cm)

confusion_matrix = pandas.DataFrame(cm, index = [i for i in ["chatgpt","human"]],
                  columns = [i for i in ["chatgpt","human"]])

plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True,cmap="YlGnBu", fmt='g')
plt.show()

print("predictions: New values")
xx_test = [' مجموع 5 و 7 و 9 يساوي 21.']
#should get 1
encode_xx_test = text_vectorization(xx_test)
pr = model_p.predict(encode_xx_test)
print(xx_test)
print(pr[0])
pp = np.argmax(pr[0])

Result = lambda a :  '1 - Chat GPT genarated' if(a > 0) else '0 - human created'
print(Result(pp))

xx_test = [ "رغم استخدام تنظيم داعش أحدث وسائل الإخراج والتصوير السينمائي، إلا أنه غفل عن خطأ كشف حقيقة الفيديو الوحشي الذي يظهر عملية نحر الرهائن المصريين الأقباط. وفي فيديو ذبح الرهائن المصريين الذي بث أوّل من أمس الأحد، يسمع بوضوح صوت المخرج بالدقيقة 3:28، يأمر ""الممثلين الدواعش"" ببدء عملية تنفيذ الإعدام باللغة الإنجليزية. ويرى خبراء أن مخاطبة المخرج للعناصر باللغة الإنجليزية، يدل أنهم ليسوا من ليبيا، ويؤكد أن أغلب مقاتلي داعش في ليبيا من المرتزقة الأجانب نظراً لضخامة أحجامهم، وعدم إجادتهم اللغة العربية، لغة القرآن الكريم. وبث تنظيم داعش في ليبيا، فيديو يظهر نحر 21 مصرياً، مستخدماً أحدث تقنيات الإخراج والتصوير باستخدام أكثر من كاميرا، وتقطيع اللقطات وتقنيات المونتاج."]
#should get 0
encode_xx_test = text_vectorization(xx_test)
pr = model_p.predict(encode_xx_test)
print(xx_test)
print(pr[0])
pp = np.argmax(pr[0])
print(Result(pp))
