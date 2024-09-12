import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.src.layers import TextVectorization 
from keras.src.models import Model
from keras.src.layers import Dense, Input, Dropout, LSTM, Activation,Embedding,Bidirectional
from keras.preprocessing import sequence  # type: ignore
from keras.initializers import glorot_uniform # type: ignore
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.src.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src import ops

@tf.keras.utils.register_keras_serializable()
class myOwn2( optimizer.Optimizer):
 
  def __init__(
        self,
        learning_rate=0.001,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="myOwn2",
        **kwargs,
  ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.pv_g = []
        self.pv_v = []
        self._learning_rate = learning_rate
        self._is_first = True
    

  def build(self, var_list):
    super().build(var_list)
    self.pv_gs = []
    self.pv_vs = []
    for var in var_list:
      self.pv_gs.append(
          self.add_variable_from_reference(
              var,
              "pv_g",
          ))
      self.pv_vs.append(
          self.add_variable_from_reference(
                var,
                "pv_v",
          ))
   
  def update_step(self, grad, variable, learning_rate):
    var_dtype = variable.dtype.base_dtype
    lr = ops.cast(learning_rate, variable.dtype)
    grad = ops.cast(grad, variable.dtype)
    new_var_m = variable - grad * lr

    pg_var = self.pv_gs[self._get_variable_index(variable)]
    pv_var = self.pv_vs[self._get_variable_index(variable)]

    if self._is_first:
        self._is_first = False
        new_var = new_var_m
    else:
        cond = grad*pg_var >= 0
        print(cond)
        avg_weights = (pv_var + variable)/2.0
        new_var = tf.where(cond, new_var_m, avg_weights)

    pv_var.assign(variable)
    pg_var.assign(grad)
    variable.assign(new_var)

  def get_config(self):
    config = super().get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter(self._learning_rate),
    })
    return config


@keras_export(["keras.optimizers.myOwn1"])
class myOwn1(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="myOwn1",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )

    def build(self, var_list):
        if self.built:
            return
        #base class build is called 
        super().build(var_list)
              
    #this is the funtion we need to implement to update waights in the layers
    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        print("update_step called")
        self.assign_sub(
            variable,
            learning_rate*gradient,
        )
        print (variable)

    def get_config(self):
        config = super().get_config()
        return config
    

def get_model(max_tokens=23, hidden_dim=64):
    inputs = Input(shape=(None,), dtype="int64")
    embedded = Embedding( input_dim=max_tokens, output_dim=256, mask_zero=True) (inputs)
    x = Bidirectional(LSTM(32))(embedded)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="softmax")(x) # binary activation output(sigmoid:1,softmax":n))
    model = keras.Model(inputs, outputs)
    #model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    model.compile(optimizer=myOwn2(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
    return model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
model.fit ( LSTM_X_train, encode_y_train,epochs=2, callbacks=callbacks)

loss, acc = model.evaluate(LSTM_X_test, encode_y_test)
print("Test accuracy = ", acc)
model.save_weights('LSTRNNMDL.weights.h5')

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
