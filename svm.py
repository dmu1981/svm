from sklearn import svm
import json
import numpy as np
import matplotlib.pyplot as plt

def draw_samples(mu, cov, n, label):
  # Cholesky decomposition
  c = np.linalg.cholesky(cov)

  # Draw random samples
  x = np.tile(mu, n) + c @ np.random.normal(size=(2, n))
  y = np.tile(np.array([label]), n)
  return x, y

def generate_set():
  samples, labels = draw_samples(mu = np.array([[3],[14]]), 
                                 cov = np.array([[2, 0.5],[0.5,3]]),
                                 n = 200,
                                 label = 1)

  s, l = draw_samples(mu = np.array([[-3],[10]]), 
                        cov = np.array([[3, -1.5],[-1.5,2]]),
                        n = 100,
                        label = 0)

  samples = np.concatenate((samples, s), axis=1)
  labels = np.concatenate((labels, l), axis=0)

  s, l = draw_samples(mu = np.array([[8],[13]]), 
                        cov = np.array([[1, 0.5],[0.5,3]]),
                        n = 100,
                        label = 0)

  samples = np.concatenate((samples, s), axis=1)
  labels = np.concatenate((labels, l), axis=0)

  s, l = draw_samples(mu = np.array([[3],[23]]), 
                        cov = np.array([[1, 0.5],[0.5,3]]),
                        n = 100,
                        label = 0)

  samples = np.concatenate((samples, s), axis=1)
  labels = np.concatenate((labels, l), axis=0)

  s, l = draw_samples(mu = np.array([[-4],[19]]), 
                        cov = np.array([[1, 0.5],[0.5,3]]),
                        n = 100,
                        label = 0)

  samples = np.concatenate((samples, s), axis=1)
  labels = np.concatenate((labels, l), axis=0)

  s, l = draw_samples(mu = np.array([[-3],[30]]), 
                        cov = np.array([[6, -1.5],[-1.5,2]]),
                        n = 200,
                        label = 1)

  samples = np.concatenate((samples, s), axis=1)
  labels = np.concatenate((labels, l), axis=0)     

  return samples, labels

def plot_set(samples, labels, clf, tit):
  minx, maxx = samples[0].min() - 1, samples[0].max() + 1
  miny, maxy = samples[1].min() - 1, samples[1].max() + 1

  x = np.linspace(minx, maxx, 800)
  y = np.linspace(miny, maxy, 800)
  X, Y = np.meshgrid(x, y)
  Z = clf.predict(np.c_[X.ravel(), Y.ravel()])
  Z = Z.reshape(X.shape)
  plt.contourf(x, y, Z, cmap=plt.cm.coolwarm, alpha=0.8)

  x = [samples[0][i] for i,l in enumerate(labels) if l==0]
  y = [samples[1][i] for i,l in enumerate(labels) if l==0] 
  plt.scatter(samples[0], samples[1], c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

  try:
    x = np.transpose(clf.support_vectors_)[0]
    y = np.transpose(clf.support_vectors_)[1]

    plt.plot(x,y, 'k+')
  except:
      pass
  plt.title(tit)
  
  


samples, labels = generate_set()
mu = np.transpose(np.mean(samples, axis=1))
std = np.transpose(np.std(samples, axis=1))
mu = np.transpose(np.tile(mu, (800,1)))
std = np.transpose(np.tile(std, (800,1)))
samples = (samples-mu)/std
plt.figure()
kernels=['linear','rbf','poly','sigmoid']

for index, kernel in enumerate(kernels):
  plt.subplot(2,2,index+1)
  if kernel=='sigmoid':
    clf = svm.SVC(kernel=kernel, C=0.1, coef0=-1.5, gamma=0.7)
    title = "sigmoid, r=-1.5, gamma=0.7"
  elif kernel=='poly':
    clf = svm.SVC(kernel=kernel, C=0.1, degree=2, coef0=1)
    title = "poly, d=2, r=1"
  elif kernel=='linear':
    clf = svm.LinearSVC(C=0.001)
    title = "linear"
  else:
    clf = svm.SVC(kernel=kernel, C=1)
    title = kernel
  clf.fit(np.transpose(samples), labels)

  plot_set(samples, labels, clf, title)

plt.show()