# 📚 Essential ML Research Papers and Book Chapters (Topic-wise)

This guide complements your YouTube/self-learning path with **must-read** academic resources and textbook references. Only critical topics are listed.

---

## 📘 Supervised Learning

### 📗 Book: *The Elements of Statistical Learning* (ESL) – Hastie, Tibshirani, Friedman
- Chapter 2 – Linear Regression
- Chapter 4 – Classification (Logistic Regression, LDA)
- Chapter 9 – Additive Models, Trees, and Related Methods
- Chapter 10 – Boosting and Additive Trees
- Chapter 7 – Model Assessment and Selection

### 📄 Paper: **A Few Useful Things to Know About Machine Learning** – Pedro Domingos  
→ [Link](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)  
- A foundational paper explaining biases, overfitting, feature engineering, and practical caveats.

### 📄 Paper: **Understanding the Bias-Variance Tradeoff** – Scott Fortmann-Roe  
→ [Link](https://scott.fortmann-roe.com/docs/BiasVariance.html)  
- Great intuitive plus mathematical coverage.

---

## 📊 Ensemble Methods

### 📄 Paper: **The Strength of Weak Learnability** – Robert Schapire (1990)  
→ [Link](https://www.cs.princeton.edu/~schapire/papers/strengthofweak.pdf)  
- The foundation of boosting theory.

### 📄 Paper: **XGBoost: A Scalable Tree Boosting System** – Chen & Guestrin (2016)  
→ [Link](https://arxiv.org/abs/1603.02754)  
- Read Sections 1–3, and skim the system design parts if doing large-scale training.

---

## 📙 Unsupervised Learning

## 📘 General & Theory

### 📗 Book: *Pattern Recognition and Machine Learning* – Christopher M. Bishop  
- **Chapter 9**: Mixture Models and the EM Algorithm  
- **Chapter 12**: Sequential Data (if doing time series clustering)

---

## 📊 Clustering

### 📄 Paper: *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN)* – Ester et al., 1996  
→ [Link](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)  
- 📌 Read Sections 1–4 for DBSCAN core concept.

### 📄 Paper: *Spectral Clustering Tutorial* – von Luxburg, 2007  
→ [Link](https://arxiv.org/abs/0711.0189)  
- 📌 Focus on Sections 1–3 (theory and intuition).

### 📄 Paper: *Clustering by Gaussian Mixture Models* – Dempster et al., 1977  
→ [Link](https://www.jstor.org/stable/2984875)  
- 📌 Key for understanding GMMs and Expectation-Maximization.

---

## 📐 Dimensionality Reduction

### 📄 Paper: *A Tutorial on Principal Component Analysis* – Lindsay Smith  
→ [Link](https://arxiv.org/pdf/1404.1100)  
- 📌 Clean explanation of PCA, eigenvectors, and variance.

### 📄 Paper: *Visualizing Data using t-SNE* – van der Maaten & Hinton  
→ [Link](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)  
- 📌 Read Sections 1–3 for understanding high-dimensional projection.

### 📄 Paper: *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction* – McInnes et al., 2018  
→ [Link](https://arxiv.org/abs/1802.03426)  
- 📌 Intro and Section 2 for comparison with t-SNE.

---

## 📦 Association Rule Learning

### 📄 Paper: *Fast Algorithms for Mining Association Rules* – Agrawal & Srikant, 1994 (Apriori)  
→ [Link](http://www.vldb.org/conf/1994/P487.PDF)  
- 📌 Read the Intro and Algorithm sections.

### 📄 Paper: *FP-Growth: Mining Frequent Patterns without Candidate Generation* – Han et al., 2000  
→ [Link](https://doi.org/10.1145/335191.335372)  
- 📌 Focus on the core algorithm and performance comparison with Apriori.

---

## ⚠️ Anomaly Detection

### 📄 Paper: *Isolation Forest* – Liu, Ting, Zhou (2008)  
→ [Link](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)  
- 📌 Sections 1–4 explain the key intuition and results.

### 📄 Paper: *A Survey of Outlier Detection Methodologies* – Chandola et al., 2009  
→ [Link](https://www.cs.umn.edu/sites/cs.umn.edu/files/tech_reports/07-017.pdf)  
- 📌 Reference guide for many techniques, including One-Class SVM.

---

## 🧠 Bonus Book (Practical Focus)

### 📘 *Hands-On Unsupervised Learning Using Python* – Ankur Patel  
- Easy-to-follow with implementation examples  
- Covers clustering, dimensionality reduction, anomaly detection

---

## 🤖 Reinforcement Learning

### 📗 Book: *Reinforcement Learning: An Introduction* – Sutton & Barto (2nd Edition)  
→ [Link (free)](http://incompleteideas.net/book/the-book-2nd.html)

- Chapter 1 – Intro
- Chapter 3 – Finite Markov Decision Processes
- Chapter 4 – Dynamic Programming
- Chapter 5 – Monte Carlo Methods
- Chapter 6 – Temporal-Difference Learning
- Chapter 9 – On-policy and Off-policy Learning (SARSA vs Q-Learning)
- Chapter 13 – Policy Gradient Methods
- Appendix: For math review

### 📄 Paper: **Playing Atari with Deep Reinforcement Learning** – DeepMind (DQN)  
→ [Link](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
- Read Abstract, Intro, and Sections 3–5 for architecture and results.

### 📄 Paper: **Proximal Policy Optimization Algorithms** – Schulman et al. (OpenAI)  
→ [Link](https://arxiv.org/abs/1707.06347)  
- Read Sections 1–3 and Appendix for core PPO logic.

---

## 🧠 Neural Networks (Cross-cutting)

### 📗 Book: *Deep Learning* – Ian Goodfellow, Yoshua Bengio, Aaron Courville  
→ [Link](https://www.deeplearningbook.org/)  
- Chapter 6 – Deep Feedforward Networks
- Chapter 7 – Regularization
- Chapter 8 – Optimization for Training
- Chapter 11 – Practical Methodology
- Chapter 10 – Sequence Modeling (RNNs)

---

## 🧰 Model Interpretability

### 📄 Paper: **A Unified Approach to Interpreting Model Predictions (SHAP)**  
→ [Link](https://arxiv.org/abs/1705.07874)  
- Read Abstract, Introduction, and Section 3 for SHAP values.

### 📄 Paper: **"Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME)**  
→ [Link](https://arxiv.org/abs/1602.04938)  
- Read Abstract and Sections 1–3.

---

## 🧪 Meta-Learning & Experimentation

### 📄 Paper: **Data Leakage in ML** – Towards Data Science (Practical Read)  
→ [Link](https://towardsdatascience.com/data-leakage-in-machine-learning-what-is-it-and-how-to-avoid-it-96f1e5efac0b)

---

### ✅ How to Use This

- Start with **book chapters** for fundamental theory
- Use **papers** when working on implementation-heavy or research-driven projects
- Combine this with **YouTube + coding practice** (e.g., Kaggle, scikit-learn, PyTorch)

---

Happy Learning & Experimenting 🚀  
Feel free to fork this roadmap and add more papers per use-case!
