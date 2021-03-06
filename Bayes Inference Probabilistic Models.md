<div style="text-align:center"><p><b><font size="6">Bayes Inference: Foundation of Machine Learning</font></p></div>

<div style="text-align:right"><p><b><font size="3">Jinhan Mei<br>May 2020</font></p></div>

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



### 1.3 Operation on Probalistic Models

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

​		首先，对我们模型大小，或者说对参数个数影响最多的是: 1. 模型中联合分布的自变量的个数；2. 模型中自变量的形态(state)。我们可以想象我们的模型为一个大小为 $k^n$ 的盒子，其中有 $n$ 个自变量，每个自变量都有 $k$ 个形态，每一个盒子里面都放着这个自变量该形态下的概率。于是乎随着 $n$ 或者 $k$ 增加的时候，我们的盒子的体积也会指数级的迅速增长。



#### 降低联合分布的维度

​		在实际的问题中，我们的模型往往是需要考虑大量的自变量来达到预测的准确性。所以给模型降维是一个我们始终绕不过的问题。其中最简单的方法就是假定各个变量之间相互独立。

​		假设我们需要解决的问题有三个自变量，每个自变量都有三个形态：$T=\{cold,\ hot,\ warm\}$, $W=\{sunny, rainy, snowy\}$, $M=\{walk,bike,bus\}$。那么这个时候我们就需要一个$3\times3\times3$的立方体来储存所有的参数。

>  **注**：其实这个描述并不完全准确，因为对于每一个变量我们都有$\sum_xP(X=x)=1$，所以我们实际需要的变量的个数要少1个。但即便如此，我们参数的个数依旧是$O(n)$，并没有变化。



​		**我们用来给模型降维最首要的方法就是假设变量之间互相独立**。



​		举例来讲：假设 $T,W,M$ 中每个形态都不是互相独立的(dependent)，那么这个时候我们有：
$$
\begin{align*}
P(T,W,M)&=P(T|W,M)P(W|M)P(M)\\
\end{align*}
$$
​		在计算 $P(T,W,M)$ 的时候，我们需要 $3\times3\times3=27$ 个参数。



​		假设$ T,W,M$中的每个变量和每个形态都互相独立(independent)，那么我们这个时候有:
$$
P(T,W,M) = P(T)P(W)P(M)
$$
​		那么在计算 $P(T,W,M)$ 的时候，我们只需要3个参数。



​		我们在上面的例子看到假设变量之间相互独立可以极大地压缩参数的个数，但是这样就会损伤我们的模型，使得它同时也会降低很多模型的表现力。但是如果我们假设所有的变量都是dependent的话，那我们就会获得一个表现力最大化的模型，但是我们需要对它每一个状态设置参数，容易使得参数爆炸或者运算时间过长。对于如何取舍这个问题我们将会不断在未来进行讨论。



#### 1.6 似然方程(Likelihood function)

​		目前我们讨论的问题一直集中在 $p(x|\theta)$ 上，我们探讨了如何通过给定的 $\theta$ 来计算对应的 $x$ 的概率。那么我们该如何获得对应数据集 $x$ 下 $\theta$ 的最优解呢？这个时候，我们便开始研究一个对于固定 $x$ 的关于 $\theta$ 的方程，也就是我们的似然方程。

​		一般来讲，我们最常见的一种似然方程是“**对数似然方程(Log Likelihood Function)**”:
$$
l(\theta;x) = log\ p(x|\theta)
$$

> 注：其实似然方程的本质是一个表达形式的转换，可以让我们更容易的在给定数据 $x$ 的情况下讨论 $\theta$ 的方程。

​		对于一个**独立同分布(Independently Identically Distribution or I.I.D.)**的数据，我们有：
$$
\begin{align*}
p(D|\theta) &= \prod_mp(x^{(m)}|\theta)\\\\
l(\theta;D)&=\sum_mlog\ p(x^{(m)}|\theta)
\end{align*}
$$


​		对于独立同分布的数据来说，他的似然方程转化成为了对每个单独的观察的乘积。但是我们发现我们**很难优化一个关于乘积的方程**，但是我们可以使用**对数函数(Log function)**，将我们的乘积方程转化称为一个求和方程，使得求解似然方程的最大值变为一个可以实际解决的问题。

> 注：除了对数似然方程，我们在解决问题中还会经常使用“**负对数似然方程(Negative Log Likelihood)**” $NLL(\theta, D)$。负对数似然函数将只在我们的对数似然函数前加了一个符号，将我们求解最大对数似然函数转化成为了求解最小负对数似然函数的问题。二者本质上并无区别。



​		在有了似然函数的加持之后，我们该如何和求解 $\theta$ 呢？反观我们自己学习的过程，我们希望我们的决策体系的决策结果和实际结果所拟合，或者说尽力做到重合。也就是说，假定我们把我们的预测结果称为 $X'$, 把实际结果称为 $X$, 那么我们希望 $\delta=|X-X'|$ 越小越好。我们“学习“的过程其实就是在学习调整我们的决策体系，缩小 $\delta$ 的过程。同理，我们在Machine Learning的过程中也是去通过调整我们的参数，使得预测的值和实际值尽可能的拟合。我们把预测的函数 $f(\theta)$ 和真实的函数 $p(\theta)$ 之间的差做某些变换后得到的函数 $L(\theta)$ 作为我们的**损失函数(Loss function)**。于是我们将求解 $\theta$ 转化成了对 $L(\theta)$ 求最小值的问题。一般来讲，我们的损失函数 $L(\theta)$ 都包含了$l(\theta)$，常见的损失函数有:


$$
\begin{align*}
最大似然函数Maximum\ Likelihood\ Estimation\ (MLE)&: L(\theta)=l(\theta;D)\\
最大后似然函数Maximum\ a\ Posterior\ (MAP)&: L(\theta)=l(\theta;D)+r(\theta)
\end{align*}
$$

##### 最大似然函数(MLE)

​		在最大似然函数中，我们只需很直觉的找到对应的 $\theta$ 使得我们的函数达到最大值即可。即：
$$
\hat{\theta}_{MLE}=arg\ \max_{\theta}l(\theta;D)
$$


#### 1.7 充分统计量(Suffcient Statistics)

​		在统计学问题中，我们采样的数据会有很多冗余的。在前文中我们讲到过在当输入的自变量越多，我们模型”学习“的参数 $\theta$ 就会越多，学习的”成本“也会更高。所以说我们希望可以在可以保留足够的信息的情况下删除所有”冗余“的数据，来达到最大限度的缩减我们的自变量的目的。

​		那么什么是”冗余“的数据呢？举例来说，对于最简单的”投硬币问题“：我们随机投掷1,000次硬币后，我们将会获得一个数据集 $T=\{H,T,T,H,T...\}$。假如我们的硬币是一个”公平硬币(Fair Coin)“，那么我们可以知道$P(H)= P(T)=\frac{1}{2}$，那么我们其实并不需要1,000个数据，我们只需2个数据的采样 $T=\{H,T\}$ 就可以准确的反映出该硬币投掷时的概率分布，剩下998个数据都可以被认为是冗余的数据。

​		我们对留下的最小的足够反应原本分布的数据，称之为**充分统计量(Suffcient Statistics)**，在数学上，我们对充分统计量 $T(X)$ 的定义为：
$$
P(\theta|T(X))=P(\theta|X)
$$


[返回上一页](Tutorial_Page.html)





