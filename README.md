# logsmldingo
#🎯 FIND S 🎯
import csv
a = []
with open('/content/sample_data/enjoysport.csv', 'r') as csvfile:
  for row in csv.reader(csvfile):
    a.append(row)
    #print(a)
    num_attribute = len(a[0]) - 1
print("\n The initial hypothesis is : ")
hypothesis = ['0'] * num_attribute
print(hypothesis)
print("\n The total number of training instances are : ",len(a))
for i in range(0, len(a)):
      if a[i][num_attribute] == 'yes':
        for j in range(0, num_attribute):
          if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
            hypothesis[j] = a[i][j]
          else:
            hypothesis[j] = '?'
      print("\n The hypothesis for the training instance {} is :\n".format(i + 1), hypothesis)
      print("\n The Maximally specific hypothesis for the training instance is ") 
      print(hypothesis)









 
🎯PCA🎯
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("/content/sample_data/winequality_red.csv")

x = dataset.iloc[:, 0:11].values 
y = dataset.iloc[:, 11].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train 
x1, x2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,stop=X_set[:, 0].max() + 1, step=0.01), np.arange(start=X_set[:, 1].min() - 1,stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75, cmap=ListedColormap(('yellow','white', 'aquamarine')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c=ListedColormap(('red', 'green','blue'))(i), label=str(j))
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
 
#🎯CANDIDATE🎯#
import numpy as np
import pandas as pd
data = pd.read_csv('/content/sample_data/enjoysport.csv')
concepts = np.array(data.iloc[:, 0:-1])
print(concepts)
target = np.array(data.iloc[:, -1])
print(target)

def learn(concepts, target):
  specific_h = concepts[0].copy()
  print("initialization of specific_h and general_h")
  print(specific_h)
  general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
  print(general_h)
  for i, h in enumerate(concepts):
      if target[i] == "yes":
        print("If instance is Positive ")
        for x in range(len(specific_h)):
          if h[x] != specific_h[x]:
            specific_h[x] = '?'
            general_h[x][x] = '?'
          
      if target[i] == "no":
        print("If instance is Negative ")
        for x in range(len(specific_h)):
          if h[x] != specific_h[x]:
            general_h[x][x] = specific_h[x]
          else:
            general_h[x][x] = '?'
      print(" steps of Candidate Elimination Algorithm",i + 1)
      print(specific_h)
      print(general_h)
      print("\n")
      print("\n")
  indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
  for i in indices:
      general_h.remove(['?', '?', '?', '?', '?', '?'])
  return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
 
🎯NAÏVE BAYSEAIN CLASSIFIER🎯
import csv
import random
import math

def loadcsv(filename):
  lines = csv.reader(open(filename, "r"));
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

def splitdataset(dataset, splitratio):
  trainsize = int(len(dataset) * splitratio);
  trainset = []
  copy = list(dataset);
  while len(trainset) < trainsize:
    index = random.randrange(len(copy));
    trainset.append(copy.pop(index))
  return [trainset, copy]

def separatebyclass(dataset):
  separated = {} 
  for i in range(len(dataset)):
    vector = dataset[i]
    if (vector[-1] not in separated):
      separated[vector[-1]] = []
      separated[vector[-1]].append(vector)
  return separated

def mean(numbers):
  return sum(numbers) / float(len(numbers))

def stdev(numbers):
  avg = mean(numbers)
  variance = sum([pow(x - avg, 2) for x in numbers]) /float(len(numbers) - 1)
  return math.sqrt(variance)

def summarize(dataset): 
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)];
  del summaries[-1] 
  return summaries
def summarizebyclass(dataset):
  separated = separatebyclass(dataset);
  summaries = {}
  for classvalue, instances in separated.items(): 
    summaries[classvalue] = summarize(instances)
  return summaries

def calculateprobability(x, mean, stdev):
  exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
  return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateclassprobabilities(summaries, inputvector):
  probabilities = {}
  for classvalue, classsummaries in summaries.items():
    probabilities[classvalue] = 1
    for i in range(len(classsummaries)):
      mean, stdev = classsummaries[i]
      x = inputvector[i]
      probabilities[classvalue] *= calculateprobability(x, mean, stdev);
      return probabilities

def predict(summaries, inputvector):
  probabilities = calculateclassprobabilities(summaries,inputvector)
  bestLabel, bestProb = None, -1
  for classvalue, probability in probabilities.items():
    if bestLabel is None or probability > bestProb:
      bestProb = probability
      bestLabel = classvalue
    return bestLabel

def getpredictions(summaries, testset):
  predictions = []
  for i in range(len(testset)):
    result = predict(summaries, testset[i])
    predictions.append(result)
  return predictions

def getaccuracy(testset, predictions):
  correct = 0
  for i in range(len(testset)):
    if testset[i][-1] == predictions[i]:
      correct += 1
  return (correct / float(len(testset))) * 100.0
def main():
  filename ='/content/sample_data/naivedata.csv'
  splitratio = 0.67
  dataset = loadcsv(filename);
  trainingset, testset = splitdataset(dataset, splitratio)
  print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingset), len(testset)))
  summaries = summarizebyclass(trainingset);
  predictions = getpredictions(summaries, testset)
  accuracy = getaccuracy(testset, predictions)
  print('Accuracy of the classifier is :{0}%'.format(accuracy))
  
🎯DECISION TREE🎯
import pandas as pd
import seaborn as sns
df = pd.read_csv("/content/sample_data/DecisionTree.csv")
df.head()
sns.countplot(x='Attrition', data=df)
df.drop(["WorkLifeBalance"], axis='columns', inplace=True)
categorical_col = []
for column in df.columns:
  if df[column].dtype == object and len(df[column].unique()) <= 50:
    categorical_col.append(column)
df['Attrition'] = df['Attrition'].astype("category").cat.codes
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for column in categorical_col:
  df[column] = label.fit_transform(df[column])
from sklearn.model_selection import train_test_split
x = df.drop('Attrition', axis=1)
y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=42)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
def print_score(clf, x_train, y_train, x_test, y_test, train=True):
  if train:
    pred = clf.predict(x_train)
    clf_report = pd.DataFrame(classification_report(y_train, pred,output_dict=True))
    print("Train Result:\n================================== ")
    print(f"Accuracy score: \n {accuracy_score(y_train, pred) * 100:.2f}%")
    print(f"Classification Report: \n {clf_report}")
    print(f"Confusion matrix: \n {confusion_matrix(y_train, pred)}\n")
  elif train == False:
    pred = clf.predict(x_test)
    clf_report =pd.DataFrame(classification_report(y_test, pred,
    output_dict=True))
    print("Test Result:\n================================= ")
    print(f"Accuracy score: \n {accuracy_score(y_test,pred) * 100:.2f}%")
    print(f"Classification Report: \n {clf_report}")
    print(f"Confusion matrix: \n{confusion_matrix(y_test, pred)}\n")

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(x_train, y_train)
print_score(tree_clf, x_train, y_train, x_test, y_test,train=True)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)
 
🎯Least Square Regression🎯
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize']= (12.0,9.0)
data=pd.read_csv('/content/sample_data/leastsquareregression.csv')
X=data.iloc[:,0]
Y=data.iloc[:,1]
plt.scatter(X,Y)
plt.show()
X_mean=np.mean(X)
Y_mean=np.mean(Y)
num=0
den=0
for i in range (len(X)):
  num+= (X[i] -X_mean)*(Y[i]-Y_mean)  
  den+= (X[i] -X_mean)**2
m=num/den
c=Y_mean - m*X_mean
print(m,c)
Y_pred=m*X +c
Y_pred=m*X+c
plt.scatter(X,Y)
plt.plot([min(X),max(X)], [min(Y_pred),max(Y_pred)],color='red')
plt.show()
 
🎯ID3🎯
import math
import csv
def load_csv(filename):
  lines = csv.reader(open(filename, "r"));
  dataset = list(lines)
  headers = dataset.pop(0)
  return dataset, headers

class Node:
  def __init__(self, attribute):
    self.attribute = attribute
    self.children = []
    self.answer = ""

def subtables(data, col, delete):
  dic = {}
  coldata = [row[col] for row in data]
  attr = list(set(coldata))
  counts = [0] * len(attr)
  r = len(data)
  c = len(data[0])
  for x in range(len(attr)):
    for y in range(r):
      if data[y][col] == attr[x]:
        counts[x] += 1
  for x in range(len(attr)):
    dic[attr[x]] = [[0 for i in range(c)] for j in range(counts[x])]
    pos = 0
    for y in range(r):
      if data[y][col] == attr[x]:
        if delete:
          del data[y][col] 
          dic[attr[x]][pos] = data[y]
        pos += 1
  return attr, dic

def entropy(S):
  attr = list(set(S))
  if len(attr) == 1:
    return 0
  counts = [0, 0]
  for i in range(2):
    counts[i] = sum([1 for x in S if attr[i] == x]) /(len(S) * 1.0)
    sums = 0
  for cnt in counts:
    sums += -1 * cnt * math.log(cnt, 2)
  return sums
def compute_gain(data, col):
  attr, dic = subtables(data, col, delete=False)
  total_size = len(data)
  entropies = [0] * len(attr)
  ratio = [0] * len(attr)
  total_entropy = entropy([row[-1] for row in data])
  for x in range(len(attr)):
    ratio[x] = len(dic[attr[x]]) / (total_size * 1.0)
    entropies[x] = entropy([row[-1] for row in dic[attr[x]]])
    total_entropy -= ratio[x] * entropies[x]
  return total_entropy
def build_tree(data, features):
  lastcol = [row[-1] for row in data]
  if (len(set(lastcol))) == 1:
    node = Node("")
    node.answer = lastcol[0]
    return node
  n = len(data[0]) - 1
  gains = [0] * n
  for col in range(n):
    gains[col] = compute_gain(data, col)
    split = gains.index(max(gains))
    node = Node(features[split])
    fea = features[:split] + features[split + 1:]
    attr, dic = subtables(data, split, delete=True)
  for x in range(len(attr)):
    child = build_tree(dic[attr[x]], fea)
    node.children.append((attr[x], child))
  return node
def print_tree(node, level):
  if node.answer != "":
    print(" " * level, node.answer)
    return
  print(" " * level, node.attribute)
  for value, n in node.children:
    print(" " * (level + 1), value)
    print_tree(n, level + 2)
def classify(node, x_test, features):
  if node.answer != "":
   print(node.answer)
  return
  pos = features.index(node.attribute)
  for value, n in node.children:
    if x_test[pos] == value:
      classify(n, x_test, features)
dataset, features =load_csv("/content/sample_data/DecisionTree.csv")
node1 = build_tree(dataset, features)
print("The decision tree for the dataset using ID3algorithm is")
print_tree(node1, 0)
testdata, features = load_csv("/content/sample_data/DecisionTree.csv")
for xtest in testdata:
  print("The test instance:", xtest)
  print("The label for test instance:", end=" ")
  classify(node1, xtest, features)
 
🎯Distance Method🎯
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def euclidean_distance(a,b):
  return sqrt(sum((e1-e2)**2 for e1,e2 in zip(a,b)))

def manhattan_distance(a,b):
  return sum(abs(e1-e2)**2 for e1,e2 in zip(a,b))

def minkowski_distance(a,b,p):
  return sum(abs(e1-e2)**p for e1,e2 in zip(a,b))**(1/p)

actual= [1,0,0,1,0,0,1,0,0,1]
predicted=[1,0,0,1,0,0,1,0,0,1]

dist1= euclidean_distance(actual,predicted)
dist2= manhattan_distance(actual, predicted)
dist3= minkowski_distance(actual,predicted,1)
dist3= minkowski_distance(actual,predicted,2)
print(dist3)

matrix= confusion_matrix(actual,predicted,labels=[1,0])
print('confusion matix :n',matrix)
tp,fn, fp,tn=confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('outcome values : \n',tp, fn, fp, tn)
matrix= classification_report(actual, predicted,labels=[1,0])
print('classification report :\n',matrix)

🎯Rule Based Method🎯
def apply_discount(age):
  if age < 18:
    return "Not eligible for a discount."
  elif 18 <= age < 30:
    return "You qualify for a 10% discount."
  elif 30 <= age < 50:
    return "You qualify for a 20% discount."
  else:
    return "You qualify for a 30% discount."
if __name__ == "__main__":
  test_cases = [15, 25, 35, 55]
  for age in test_cases:
    result = apply_discount(age)
    print(f"Age: {age} - {result}")
 
🎯Locally Weighted Non-Parametric🎯
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
  m, n = np.shape(xmat)
  weights = np.mat(np.eye((m)))
  for j in range(m):
    diff = point - X[j]
    weights[j, j] = np.exp(diff * diff.T / (-2.0 * k **2))
  return weights

def localWeight(point, xmat, ymat, k):
  wei = kernel(point, xmat, k)
  W = (X.T * (wei * X)).I * (X.T * (wei * ymat.T))
  return W

def localWeightRegression(xmat, ymat, k):
  m, n = np.shape(xmat)
  ypred = np.zeros(m)
  for i in range(m):
    ypred[i] = xmat[i] * localWeight(xmat[i], xmat,ymat, k)
  return ypred
data = pd.read_csv('/content/sample_data/nonpara.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))
ypred = localWeightRegression(X, mtip, 0.5)
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red',
linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show();





🎯ANN using backpropagation🎯
import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)
y = y / 100
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def derivatives_sigmoid(x):
  return x * (1 - x)
epoch = 5000 
lr = 0.1 
inputlayer_neurons = 2 
hiddenlayer_neurons = 3
output_neurons = 1 
wh = np.random.uniform(size=(inputlayer_neurons,
hiddenlayer_neurons)) 
bh = np.random.uniform(size=(1, hiddenlayer_neurons)) 
wout = np.random.uniform(size=(hiddenlayer_neurons,
output_neurons)) 
bout = np.random.uniform(size=(1, output_neurons))
for i in range(epoch):
  hinp1 = np.dot(X, wh)
  hinp = hinp1 + bh
  hlayer_act = sigmoid(hinp)
  outinp1 = np.dot(hlayer_act, wout)
  outinp = outinp1 + bout
  output = sigmoid(outinp)
  EO = y - output
  outgrad = derivatives_sigmoid(output)
  d_output = EO * outgrad
  EH = d_output.dot(wout.T)
  hiddengrad = derivatives_sigmoid(hlayer_act)
  d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output) * lr
wh += X.T.dot(d_hiddenlayer) * lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)

 
🎯Naviebay🎯
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

documents = ["This is first document.", "this is second document.", "this is third document.","this is fourth document.", "this is fifth document.", "this is sixth document."]
labels = [1, 0, 1, 0, 1, 0] 
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2,random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)
predictions = nb_classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions,zero_division=1))






