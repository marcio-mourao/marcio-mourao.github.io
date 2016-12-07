from sklearn import naive_bayes
import numpy as np

#The target
y = np.array((0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1))

#continuous data
x11 = np.random.normal(0,1,10)
x12 = np.random.normal(1,1,10)
x1 = np.hstack((x11,x12))
x21 = np.random.normal(10,20,10)
x22 = np.random.normal(0,10,10)
x2 = np.hstack((x21,x22))
xcont = np.transpose(np.vstack((x1,x2)))

#categorical data
cat11 = np.random.binomial(1,0.2,10)
cat12 = np.random.binomial(1,0.5,10)
cat1 = np.hstack((cat11,cat12))
cat21 = np.random.binomial(1,0.8,10)
cat22 = np.random.binomial(1,0.5,10)
cat2 = np.hstack((cat21,cat22))
xcat = np.transpose(np.vstack((cat1,cat2)))

#Model for continuous data
cont = naive_bayes.GaussianNB()
cont.fit(xcont,y)
print(cont.score(xcont,y))
pcont=cont.predict_proba(xcont)

#Model for categorical data
cat = naive_bayes.MultinomialNB()
cat.fit(xcat,y)
print(cat.score(xcat,y))
pcat = cat.predict_proba(xcat)

#Combined model
ptot = np.hstack((pcont,pcat))
comb = naive_bayes.GaussianNB()
comb.fit(ptot,y)
print(comb.score(ptot,y))

