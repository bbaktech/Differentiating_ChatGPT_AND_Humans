import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import keras
from keras.layers import TextVectorization
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,Embedding,Bidirectional
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

def get_model(max_tokens=23, hidden_dim=64):
    inputs = Input(shape=(None,), dtype="int64")
    embedded = Embedding( input_dim=max_tokens, output_dim=256, mask_zero=True) (inputs)
    x = Bidirectional(LSTM(32))(embedded)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="softmax")(x) # binary activation output(sigmoid:1,softmax":n))
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    return model

df1 = pandas.read_csv('human_chatgpt_genarated_dataset.csv')
X = df1.iloc[:, 0]
Y  = df1.iloc[:, 1]

print ('DataSet Size:'+str(len(X))+ ' Records')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

#output sequence length max_length = 600 
max_length = 600 
max_tokens = 20000

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
model.fit ( LSTM_X_train, encode_y_train,epochs=5, callbacks=callbacks)

loss, acc = model.evaluate(LSTM_X_test, encode_y_test)
print("Test accuracy = ", acc)
model.save_weights('LSTRNNMDL.h5')

#code tyo genarate confusion_matrix
y_pred = []
pr = model.predict(LSTM_X_test)
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
pr = model.predict(encode_xx_test)
print(xx_test)
print(pr[0])
pp = np.argmax(pr[0])

Result = lambda a :  '1 - Chat GPT genarated' if(a > 0) else '0 - human created'
print(Result(pp))

xx_test = [ "رغم استخدام تنظيم داعش أحدث وسائل الإخراج والتصوير السينمائي، إلا أنه غفل عن خطأ كشف حقيقة الفيديو الوحشي الذي يظهر عملية نحر الرهائن المصريين الأقباط. وفي فيديو ذبح الرهائن المصريين الذي بث أوّل من أمس الأحد، يسمع بوضوح صوت المخرج بالدقيقة 3:28، يأمر ""الممثلين الدواعش"" ببدء عملية تنفيذ الإعدام باللغة الإنجليزية. ويرى خبراء أن مخاطبة المخرج للعناصر باللغة الإنجليزية، يدل أنهم ليسوا من ليبيا، ويؤكد أن أغلب مقاتلي داعش في ليبيا من المرتزقة الأجانب نظراً لضخامة أحجامهم، وعدم إجادتهم اللغة العربية، لغة القرآن الكريم. وبث تنظيم داعش في ليبيا، فيديو يظهر نحر 21 مصرياً، مستخدماً أحدث تقنيات الإخراج والتصوير باستخدام أكثر من كاميرا، وتقطيع اللقطات وتقنيات المونتاج."]
#should get 0
encode_xx_test = text_vectorization(xx_test)
pr = model.predict(encode_xx_test)
print(xx_test)
print(pr[0])
pp = np.argmax(pr[0])
print(Result(pp))
