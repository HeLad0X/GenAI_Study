from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer(binary=True)
y_train = twenty_train.target
X_train = count_vect.fit_transform(twenty_train.data).toarray()

n = X_train.shape[0]
d = X_train.shape[1]
K = len(set(y_train))

psis = np.zeros([K,d])
phis = np.zeros([K])

for k in range(K):
    X_k = X_train[y_train == k]
    psis[k] = np.mean(X_k, axis=0)
    phis[k] = X_k.shape[0] / float(n)

print(phis)

for w in ["opengl", "gpu", "church", "rome", "fever"]:
    print(w, w in count_vect.vocabulary_)


def nb_predictions(x, psis, phis):
    n, d = x.shape
    x = x.reshape(1, n, d)          # (1,n,d)
    psis_ = psis.reshape(K, 1, d)   # (K,1,d)

    psis_ = psis_.clip(1e-12, 1 - 1e-12)

    logpy = np.log(phis).reshape(K, 1)  # (K,1)

    logpxy = x * np.log(psis_) + (1 - x) * np.log(1 - psis_)  # (K,n,d)
    logpyx = logpxy.sum(axis=2) + logpy                        # (K,n)

    return logpyx.argmax(axis=0), logpyx


idx, logpyx = nb_predictions(X_train, psis, phis)

docs_new = ['OpenGL on the GPU is fast', 'The church of rome is very old', "You should rest well when you have fever"]

X_new = count_vect.transform(docs_new).toarray()
predicted, logpyx_new = nb_predictions(X_new, psis, phis)

for doc, category in zip(docs_new, predicted):
    print(f"{doc} => {twenty_train.target_names[category]}")