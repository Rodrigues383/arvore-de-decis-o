import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("F:\CURSOS\PÓS GRADUAÇÂO\IA\show.csv")

print(df)

d = {'Sim': 1, 'Nao': 0}
df['Recomendacao'] = df['Recomendacao'].map(d)

print(df)

features = ['Idade','NumFilhos','TemCarro','TemCasaPropria']


X = df[features]
y = df['Recomendacao']



#print(X)
#print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features,filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

#img=pltimg.imread('mydecisiontree.png')
#imgplot = plt.imshow(img)
#plt.show()

print(dtree.predict([[40, 2, 1, 0]]))
print(dtree.predict([[18, 0, 0, 0]]))
print(dtree.predict([[18, 3, 1, 1]]))
print(dtree.predict([[26, 2, 1, 1]]))
