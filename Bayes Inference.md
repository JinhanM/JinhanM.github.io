<div style="text-align:center"><p><font size="5">Bayes Inference: Foundation of Machine Learning</font></p></div>



### Preface

​		To whom want to learn about Machine Learning, you may stuck at where to begin with, or you may get confused with the first chapter of a Machine Learning problem, like "Why the hell I am learning this?" or "Yes I understood the statistics you talked about here, but how is this related to Machine Learning, or How it may help my project?". So here we go, I hope we can discuss about the foundation of Machine Learning, Bayes Inference, and get a inuition of "How really Machine Learning works" at the end of this tutorial. Now, let's get started.



#### A glimpse on the core of Machine Learning

​		Before all, let's focus on "What Machine Learning really is" first. Machine Learning is developed to solve  problems that can not be solved, or really poorly solved with ordinary algorithms. Machine Learning has nothing to do with the fancy "intelligence" in the movie, it can all go down to really simple problems. 



​		For example, suppose we have a group of random variables: $(X_1,X_2,X_3,...,X_N)$ observed, or unobserved. We want a model  $p_{\theta}$ that capture all the relationship between variables. The approach of probabilistic generative models is to relate all variables by a learned **joint probability distribution** $p_\theta(X_1,X_2,...,X_N)$. Intuitively, consider our $p_\theta$, it should be a function that counts the impact of every point from $(X_1,...,X_{k-1},X_{k+1},...,X_N)$ on $X_k$. Is the function good enough? No. We neeeds to count the impact of correlation of $(X_1,X_2)$ on $X_k$, also $(X_1,X_2,X_3)$ on $X_k$. Adding all the correlation up, we will get exactly a **joint probablity distribution**, and that is what we are looking for.



​		Assume the variables were generated by some distribution $(X_1,...,X_N)~ p_\star(X)$. "Learning" the joint probability distribution, also called **density estimation** is the process choosing the parameters $\theta$ of a specified parametric joint distribution $p_{\theta}(X)$ to "best match" the "real" $p_\star(X)$ . 



​		**To achieve the goal, we have three problems that we focusing to solve:**



* How should we specify $p_\theta$? or What should $p_\theta$ should look like? Is there any other way we can represent the $p_\theta$? (Because we would have infinite parameters  brutal joint probablity distribution)
* How can we make sure that $p_\theta$ is the "Best Match"? or What is the meaning of "Best Match"?
* How can we find the best parameters of $\theta$ ?





## Chapter 1: Introduction to Probabilistic Models of Machine Learning

​		With representing the model by joint probability distribution, we can think about common machine learning tasks differently, where random variables represent:

* Input data X

* Discrete output or "Labels" C

  or

  Continous output Y



​		Then we have the joint probability distribution over these random variables,$p(C,X)$ or $p(X,Y)$, we see that this can be used for Machine Learning Tasks like:

	1. Regression:  $p(Y|X)=\frac{p(X,Y)}{p(X)} = \frac{p(X,Y)}{\int p(X,Y)dy}$
 	2. Classification / Clustering:  $p(C|X) = \frac{p(X,C)}{\sum_Cp(X,C)}$



#### Classification VS Clustering : Oberserved VS Unobserved Random Variables

​		The distinction between Classification VS Clustering, or Supervised vs Unsupervised Learning, is given by whether a random variable is **oberseved** or **unobserved**. For example:



​		**Supervised Dataset**:$\{x_i,c_i\}_{i=1}^N \sim p(X,C)$

​		In this case, the class labels are "observed", and we are looking for finding the conditional distribution $p(C|X)$ satisfies the supervised classification problem.



​		However, we may encounter datasets that only contains the "input data":



​		**Supervised Dataset**:$\{x_i\}_{i=1}^N \sim p(X,C)$

​		Notice that we did not change the generative assumption, our data $x_i$ is still distributed according to a class label $C=c_i$, even though it is **unobserved** in the dataset. The common way to refer to an unobserved discrete class label is "cluster".



​		However, whether the label is observed or not, **it does not ultimately change our goal**, which is to have a model of the conditional distribution over the labels/clusters given the input data $p(C|X)$.



​		This view allows us to easily accommodate **semi-supervised** learning, when variables are observed for some, but not all, of the examples in the dataset.



​		We notice a interesting thing is that in practice, there may be some variables that we can not dierectly observed, but they still play an important roll in our model. For example, suppose we want to build a ranking system, we have winning and losing record for 10,000 games between 100 players for our data. We can only observe the "result" of a game between two players, but we can not observe the "skills" of a player. Intuitionally, we will use the result of games to measure the "skills" of a player, and then predict their "winning rate" using the "skills". Variables like "skills" that can not observed directly is called **Latent Variable**.



#### Latent Variables

​		Further, like clusters, introducing assumptions about unobserved variables is a powerful modelling tool. We will make use of this by modelling variables which are **never observed** in the dataset, called latent or hidden variables. By introducing and modelling latent variables, we will be able to naturally describe and capture abstract features of our input data.



### 1.3 Operation on Probailistic Models

​		To complete a Machine Learning, we will perform several operations on a probablistic model:	

* **Generate Data** : For the first step, we need to know how to **sample** from the model.
* **Estimate Likelihood**: When all variables are either observed or marginalized the result is a single real number which is the probability of the all variables taking on those specific values.
* **Inference**: Compute expected value of some variables given others which are either observed or marginalized.
* **Learning**: Set the parameters of the joint distribution given some observed data to maximize the probability the observed data.



#### Modeling中的重点和难点

​		在我们确定好我们将使用联合分布(Joint Distribution)来对我们的问题进行建模之后，我们将把我们的目光聚集在如下的两个问题上：

* 问题一：我们怎么可以确保我们的模型**准确计算或拟合**我们数据？

* 问题二：我们如何确保我们模型在输入的变量很多的时候依旧可以**使用尽可能少的参数(compact representation)**？

  ​		我们注意到在之前我们提到过的对于一个$\{X_1,X_2,...,X_N\}$的联合分布中，假如$X_1,...,X_N$都不是互相独立的(dependent)，那么我们需要为**每一个**相关关系(correlation)安排一个参数值，即，我们需要$O(N^N)$的参数值。这意味着假如我们的自变量的个数一旦稍微增加，我们的模型的大小和运算次数会急速增加。



​		于是乎，为了能解决这两个终极问题，我们需要对我们的模型进行一些假设。其中最重要的一条便是我们需要假设其中部分的变量**相互独立**，这将会对我们后面的的**因式分解(factorizations)**和**联合分布(joint distribution)**起到很大的作用。我们将会对这个问题在之后的章节进行具体的讨论。



### 1.4 参数化分布(Parameterized Distributions)

​		书接上文，我们刚刚讲到了我们在机器学习建模中需要做的终极目标，那么我们现在简单介绍一下如何计算分布，以及一些总结。不过今天老子不想具体介绍这个，暂且挖个坑，先放上总结。

##### 参数化分布的总结

* 假设我们现在拥有一个联合分布，那么我们可以计算它的边际分布(Marginal Distribution), 以及它的条件分布(Conditional Distribution)。
* 在我们确定一类分布之后，我们可以把这个分布简化成它的参数。
* 我们可以用一个数组或矩阵(Arrays or Matrix)存储和表示它的参数。
* 我们可以通过操作我们的数组来达到计算边际分布和条件分布的目标。



### 1.5 联合分布的维度(Dimensionality)

​		在1.3中，我们在Modeling的难点和重点中简单的对参数爆炸(问题二)进行了讨论。我们将在本小节中对这个问题进行更加具体的讨论。












