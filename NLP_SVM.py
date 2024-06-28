import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#from numpy import vectorize
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


lg = LogisticRegression(penalty='l1',solver='liblinear')
dtc = DecisionTreeClassifier(max_depth=5)
sv = SVC(kernel='sigmoid',gamma=1.0)
rfc = RandomForestClassifier(n_estimators=50,random_state=2)
gbc = GradientBoostingClassifier(n_estimators=50,random_state=2)

def prediction(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    pr = model.predict(X_test)
    acc_score = metrics.accuracy_score(y_test,pr)
    f1= metrics.f1_score(y_test,pr,average="binary", pos_label=1)
    return acc_score,f1

acc_score = {}
f1_score={}


df1 = pd.read_csv("human_chatgpt_genarated_dataset.csv")

X=df1['text']
y=df1['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Predict the test dataset and get a confusion matrix.
#This confusion matrix is basically a table that defines the performance of the algorithm.
#Here for our classification problem, it will give four values. False Positive (FP), True Positive (TP), False Negative (FN), 
#and True Negative (TN) are the four values that it will give. We will also plot this confusion matrix.

#LogisticRegression
lg.fit(X_train_tfidf, y_train)
y_pred =lg.predict(X_test_tfidf)
acc_score = metrics.accuracy_score(y_test,y_pred)
f1= metrics.f1_score(y_test,y_pred,average="binary", pos_label=1)
print('LogisticRegression:   Accuracy:' + str(acc_score) + '  F1 Score:'+ str(f1))

#DecisionTreeClassifier
dtc.fit(X_train_tfidf, y_train)
y_pred =dtc.predict(X_test_tfidf)
acc_score = metrics.accuracy_score(y_test,y_pred)
f1= metrics.f1_score(y_test,y_pred,average="binary", pos_label=1)
print('DecisionTreeClassifier:   Accuracy:' + str(acc_score) + '  F1 Score:'+ str(f1))

#RandomForestClassifier
rfc.fit(X_train_tfidf, y_train)
y_pred =rfc.predict(X_test_tfidf)
acc_score = metrics.accuracy_score(y_test,y_pred)
f1= metrics.f1_score(y_test,y_pred,average="binary", pos_label=1)
print('RandomForestClassifier:   Accuracy:' + str(acc_score) + '  F1 Score:'+ str(f1))

#GradientBoostingClassifier
gbc.fit(X_train_tfidf, y_train)
y_pred =gbc.predict(X_test_tfidf)
acc_score = metrics.accuracy_score(y_test,y_pred)
f1= metrics.f1_score(y_test,y_pred,average="binary", pos_label=1)
print('GradientBoostingClassifier:   Accuracy:' + str(acc_score) + '  F1 Score:'+ str(f1))

#SVM used
sv.fit(X_train_tfidf, y_train)
y_pred =sv.predict(X_test_tfidf)
acc_score = metrics.accuracy_score(y_test,y_pred)
f1= metrics.f1_score(y_test,y_pred,average="binary", pos_label=1)
print('Support Vector model(SVM):   Accuracy:' + str(acc_score) + '  F1 Score:'+ str(f1))

cm = confusion_matrix(y_test, y_pred)
print('confusion_matrix')
print(cm)

confusion_matrix = pd.DataFrame(cm, index = [i for i in ["chatgpt","human"]],
                  columns = [i for i in ["chatgpt","human"]])

plt.figure(figsize = (10,7))
sn.heatmap(confusion_matrix, annot=True,cmap="YlGnBu", fmt='g')
plt.show()

Result = lambda a :  '(1) - Chat GPT genarated' if(a > 0) else '(0) - human created'

#human writen text(0)
input=['خبارنا المغربية ـ الرباط علمت أخبارنا ، أن البرلماني و رجل الأعمال المعروف ميلود الشعبي قدم استقالته اليوم الاثنين ، من مجلس النواب، لأسباب صحية منعته من الاستمرار في العمل السياسي. و قد قبل مجلس النواب استقالة الشعبي ، المنتمي لحزب البيئة والتنمية المستدامة، والذي كان يرأس مجموعة نيابية مكونة من خمسة برلمانيين. ومن المرتقب أن يعوضه المرشح الثاني في اللائحة التي تقدم بها حزبه في انتخابات 25 شتنبر 2011 بعد اعلان شغور المقعد من طرف المحكمة الدستورية.']
vect_input=vectorizer.transform(input)
pr = sv.predict(vect_input)
print (input[0]+Result(pr))

#chatgpt genarated text(1)
input=['قالَ السَماءُ كَئيبَةٌ وَتَجَهَّما قُلتُ اِبتَسِم يَكفي التَجَهّمُ في السَماقالَ الصِبا وَلّى فَقُلتُ لَهُ']
vect_input=vectorizer.transform(input)
pr = sv.predict(vect_input)
print (input[0]+Result(pr))

#chat gpt genarated text(1)
input=['- تذكيرات صحية متنقلة']
vect_input=vectorizer.transform(input)
pr = sv.predict(vect_input)
print (input[0]+Result(pr))

#human writen text(0)
input=["أخبارنا المغربية ــ د ب أ أعلن تنظيم داعش في مدينة برقة الليبية، اليوم الخميس، عن إعدام صحافيين اثنين من تونس، كان خطفا منذ أكثر من شهر في ليبيا. وأعلن التنظيم على صفحته بموقع التواصل الاجتماعي ""فيس بوك"" تحمل اسم المكتب الإعلامي في برقة، عن إعدام الصحافيين سفيان الشورابي ونذير القطاري. وقال التنظيم إنه تم ""تنفيذ حكم الله على إعلاميين في فضائية محاربة للدين مفسدة في الأرض""، وأكد عدد من وسائل الإعلام التونسية خبر إعدام الصحافيين المحتجزين في ليبيا. ولم يصدر أي تأكيد أو نفي لهذه الأنباء حتى الآن من جهات رسمية. واحتجز الصحافي سفيان الشورابي والمصور نذير القطاري في مدينة برقة الليبية، عندما كانا في مهمة إعلامية في الثالث من سبتمبر (أيلول) الماضي. وفشلت السلطات التونسية منذ ذلك الحين في القيام بمفاوضات مع جهات محددة للإفراج عنهما."]
vect_input=vectorizer.transform(input)
pr = sv.predict(vect_input)
print (input[0]+Result(pr))