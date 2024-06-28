import numpy as np
import tensorflow as tf
import keras
from keras.layers import TextVectorization
from keras.models import Model ,load_model 
from keras.layers import Dense, Input, Dropout, Flatten,LSTM, Activation,Embedding,Bidirectional,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical

def get_model(max_tokens=23, hidden_dim=64):
    inputs = Input(shape=(None,), dtype="int64")
    embedded = Embedding( input_dim=max_tokens, output_dim=256, mask_zero=True) (inputs)
    x = Conv1D(16, 24, activation="relu")(embedded)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 12, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

df1 = pandas.read_csv('human_chatgpt_genarated_dataset.csv')
X = df1.iloc[:, 0]
Y  = df1.iloc[:, 1]
print ('DataSet Size:'+str(len(X))+ ' Records')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

text_vectorization = TextVectorization(ngrams=2,max_tokens=20000,output_mode="int",
                                       output_sequence_length=600,
                                       )
text_vectorization.adapt(X)
print ( 'vocabulary size:' + str(len(text_vectorization.get_vocabulary())))

encode_X_train = text_vectorization(X_train)
encode_X_test = text_vectorization(X_test)
encode_y_train = to_categorical(Y_train)
encode_y_test = to_categorical(Y_test)

model = get_model(max_tokens = len(text_vectorization.get_vocabulary()))
model.summary()
callbacks = [keras.callbacks.ModelCheckpoint("NLP_CNN2.keras",save_best_only=True)]
history = model.fit(encode_X_train,encode_y_train, epochs=5,callbacks=callbacks)
model.save_weights('NLP_CNN2.h5')
print(f"Test loss and acc: {model.evaluate(encode_X_test,encode_y_test)}")

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
