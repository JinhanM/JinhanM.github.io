## SVM 支持向量机：硬间隔、软间隔、L1正则、L2正则以及核方法的讨论及代码实现

对SVM问题的一瞥



### SVM他是用来解决哪类问题的?我们为什么需要SVM？

​	在人工智能和大数据分析的学科中，我们经常需要把数据进行进行分类，亦或是需要在数据之间找到一个最合适的分界线，来决定两类数据。举例来说，假如说我们收集了14个人的BMI值进行分析，我们希望在数据中间**找到最合适的一个的值**，**并以此为界来判断一个人是否肥胖**。假定我们收集的数据如图：

<img src="/img/SVM_Split_data_1d.png" alt="hi" style="zoom:50%;" />

​	假如说我们寻找到的**决定边界(decision boundary)**是上图中所示橙线，那么所有在橙线右边的点我们都可以认定为”肥胖“，所有在橙线左侧的点我们都可以认定为”正常“。假如说我们收集到了一个新的数据并把它标记成为黑点，我们发现黑点在橙线右侧，由此我们推论出这个人是”肥胖“的。

​	**当我们考虑如何去放置这个橙色的线的时候，我们发现其实他只要放在绿点和橙点的中间，在任意位置都是可以的。那么是否存在一个最适合的橙色的线呢？或者说我们该如何去寻找一个最合适的“界”呢？**

​	**这便是SVM算法真正的核心所在**。



### 从最简单问题入手：硬间隔向量机(Hard Margain SVM)

​	假设，我们的数据集是两堆完全**没有重合**的点，并且**线性可分**，那么这个时候我们就可以用最简单的**硬间隔向量机(Hard Margain SVM)**解决此类问题。假如我们的数据如下图所示：

<img src="/img/Figure_1.png" alt="hi" style="zoom:36%;" />

​	那么现在我们的问题来了，对于这种情况我们该如何分离两类点并且找到最优的解呢？

​	假设我们现在定义一条线 $ L $ 作为我们的决定边界：
$$
L: w^tx+b = 0
$$
​	我们同时定义两条平行于 $L$ 的线，$L1, L2$ 使得 $L1$ 经过到 $L$ 最近的蓝点，并且$L2$ 经过到 $L$ 最近的绿点。我们称为 $L1, L2$ 为我们的**支持向量(Support Vector)**
$$
L1:w^tx+b=1\\
L2:w^tx+b=-1\\
$$
​	现在我们尝试一些不同的 $L$, 然后进行观察：	

<img src="/img/Figure_2.png" alt="hi" style="zoom:36%;" />

​	假设我们现在用第一组$w, b$, 获得了第一个决定边界如上图

<img src="/img/Figure_3.png" alt="hi" style="zoom:36%;" />

​	我们现在尝试变动一下我们的$w, b$的值，我们获得了第二个决定边界。通过观察我们发现**当$L1, L2$之间的距离越大的时候，我们的两类点被决策边界分的越开，我们的决策边界也就越趋近于最优解**

​	**换言之，我们将寻找“界”的问题变成了一个寻找$L1,L2$距离最大值的数学问题**

​	我们注意到 $L$ 到 $L1$ 与 $L2$ 的距离相等，也就是说我们可以转化我们的问题为“寻找$L,L1$最大距离的问题”，即：

$$
\begin{align*}
\max{|L\ L1|}&=\max{\frac{|w^Tx+b|}{||w||}}\\
&=max\frac{1}{||w||}\\\\
&=min{||w||}
\end{align*}
$$
​	同时，我们对这个最值问题也有约束(constraint), 我们需要用这个约束把所有的蓝点($y_i=1$), 和绿点($y_i=-1$)分开。即：
$$
w^Tx_i+b\geq+1, when\ y_i=+1\\
w^Tx_i+b\leq-1, when\ y_i=-1\\
$$
​	化简可得：
$$
y_i(w^Tx_i)\geq1
$$
​	注: 化简的过程其实很简单，但第一次想可能还是会有些困惑，不妨带几个数就可以想明白。



​	**到这一步，我们把这个寻找界的问题已经化简成为了一个在约束内寻找最值的纯数学问题，而这类问题可以被Python CVXPY包内的函数所解，也就是我们现在可以上手尝试代码实现我们的SVM了。**





### 硬间隔SVM的代码实现

#### 		数据定义

​			在我们开始正式进行我们的代码实现前，我们需要先明确我们的数据输入。我们数据为X和y，其中X是一个numpy.ndarray, X中包含了所有点的坐标；y是一个list，其中用1和-1来区分对应X中每个点类的分类，1位蓝点，-1位绿点。

​			我们现在开始为我们的硬间隔SVM生成一个玩具集:

```python
import numpy as np
np.random.seed(5)

# 我们先生成以(3,3) 和 (-3,-3)为中心各生成40个坐标
x1 = np.random.normal(3, 1, (2, 40))
x2 = np.random.normal(-3, 1, (2, 40))
plt.scatter(x1[0, :], x1[1, :], color='blue')
plt.scatter(x2[0, :], x2[1, :], color='green')

def join_data(dat1, dat2):
    lst = []
    for i in range(len(dat1[0])):
        lst.append([dat1[0][i], dat1[1][i]])
    for j in range(len(dat2[0])):
        lst.append([dat2[0][j], dat2[1][j]])
    return np.asarray(lst)

# 然后我们把生成好的坐标化为我们所需的标准输入格式
X = join_data(x1, x2)
y = [1] * len(x1[0]) + [-1] * len(x2[0])
```



#### 编写硬间隔SVM

​	经过我们上面的讨论，我们最后把硬间隔SVM化成了以下问题：
$$
min{||w||},\ with:\ y_i(w^Tx_i)\geq1
$$
​	在使用CVXPY包的情况下，这个问题可以被转化成：

```python
import cvxpy as cp
import matplotlib.pylab as plt

D = X.shape[1]
N = X.shape[0]
W = cp.Variable(D)
b = cp.Variable()

# 计算损失函数loss还有约束constraint。
loss = cp.sum_squares(W)
constraint = [y[i]*(X[i]*W+b) >= 1 for i in range(len(y))]
prob = cp.Problem(cp.Minimize(loss), constraint)
prob.solve()

# 转换w和b的值
w = W.value
b = b.value

# 画出决定边界
x = np.linspace(-4, 4, 20)
plt.plot(x, (-b - (w[0] * x)) / w[1], 'r', label="decision boundary for hard SVM")
plt.legend()
plt.show()
```

​	经过以上代码，你应该可以得到一个如下图的决定边界。

<img src="/img/hard_margain_SVM_toy_data.png" alt="hi" style="zoom:36%;" />



#### 硬间隔SVM的局限：假如数据有异常值，或者两组点有重合或者略微的相交？

​	在硬间隔SVM中，我们的决策边界把所有的点都**完全正确**的分开，虽然这个方法能很“准确”的分开所有的点，但是该方法**对异常值会非常敏感**。举例来说，假如我们在我们的玩具数据中加入一个异常值，并再用硬间隔SVM对两类点进行区分，那么我们得到的决策边界(如下图)将**有悖于常识**的认知。

<img src="/img/Hard_margain_with_outlier.png" alt="hi" style="zoom:36%;" />



#### 对硬间隔支持向量机缺陷的弥补：软间隔支持向量机或L1正则支持向量机(Soft Margin SVM or L1 regularzation SVM)

​	我们在之前的图像中很好的感受到了异常值会如何影响我们的硬间隔SVM，那么我们现在就要考虑该如何解决异常值，或者是有重合的点对硬间隔SVM的影响。当我们重新考虑上图中我们的决定边界该如何定义时，我们发现最优的决定边界理应还是忽略唯一的异常值的硬间隔SVM的决策边界。于是，我们可以设计一种方法，**使其允许一些点被错误的标记，并在损失函数中对其进行惩罚，以至于达到一个相对的平衡**。假设我们对标记错误的点进行 C 倍的惩罚，当C  变小时，我们的算法会相对更少地惩罚错误标记的点，意味着可能会有更多的点被错误的标记；当C 变大时，我们的算法会更多地惩罚错误标记的点，意味着可能会有更少的点被错误标记；当 C 足够大之后，我们的算法得到的决定边界会和硬间隔SVM得到的决定边界一样。我们将这种允许错误标记点的算法称为**软间隔支持向量机(Soft Margain SVM)**。



​	在数学上，我们可以将以上的思想转化成为以下的最优化问题：
$$
Minimize\ \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\epsilon_i\\
subject\ to:\ y_i(w^Tx_i+b)\geq1-\epsilon_i,\ where\ \epsilon_i\geq0.
$$



#### 关于软间隔向量机损失函数的讨论

​	我们注意到在上面相较于硬间隔向量机，软间隔向量机增加了一项$C\sum_{i=1}^{n}\epsilon_i$， 其中我们对于 $\epsilon$ 的定义如下：
$$
\epsilon_i=\max\{0,1-y_i(w^Tx_i+b)\}
$$
​	我们注意到，凡是被正确分类的点，我们都会有：
$$
y_i(w^Tx_i+b)\geq0
$$
​	于是乎，我们就得到了一个关于 $\epsilon $ 的特别的损失方程，这个损失方程的特点在于它只会惩罚被错误分类了的点，并且它的图像看起来很像一个铰链(hinge)，于是乎我们称之为**"hinge loss"**.

<img src="/img/hinge_loss.png" alt="hi" style="zoom:36%;" />



#### 软向量机的代码实现

```python
D = X.shape[1]
N = X.shape[0]
W = cp.Variable(D)
b = cp.Variable()

# 在这里我们使用 C = 1/4
C = 1/4
loss = (0.5 * cp.sum_squares(W) + C * cp.sum(cp.pos(1 - cp.multiply(y, X * W + b))))
# 在这里我们改变了损失函数，并且移除了对w和b的约束
prob = cp.Problem(cp.Minimize(loss))
prob.solve()

# 转换w和b的值
w = W.value
b = b.value

# 画出决定边界
x = np.linspace(-4, 4, 20)
plt.plot(x, (-b - ((w[0]) * x)) / w[1], 'r', label="decision boundary for Soft Margain SVM")
plt.legend()
plt.show()
```

<img src="/img/decision_boundary_of_soft_svm_c=0.25.png" alt="hi" style="zoom:36%;" />

​	在上面的图中，我们可以看到我们的决定边界相比与有异常值的硬间隔SVM有了大幅度的提升，并且和最初没有异常值的硬间隔SVM的决定边界很接近，由此可见**软间隔向量机可以大幅降低异常值对模型的伤害。



#### Do we have another way to approach the optimization problem? Introducing Soft Margain Dual SVM.

​	Now we go back and consider the same optimization problem:
$$
Minimize\ \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\epsilon_i\\
subject\ to:\ y_i(w^Tx_i+b)\geq1-\epsilon_i,\ and\ \epsilon_i\geq0.
$$
​	**Note that we have TWO constraints here, missing one of the second one would leads to a missing part of constraint in solving Larange Function. **

​	Is there another way to solve this problem? The answer is yes. Minimizing a function $f(x)$, giving a constraint $g(x)$, is a very important problem in math. One of the solving method is Lagrange Multiplier.



#### Larange Mulitplier Method

​	As you may consider what a Larange Muliplier Method is, It is basicly is considering a problem as follows:
$$
maximizing\ f(x)\\
subject \ to\ :\ g(x)=0
$$
​	We would have a Larange function:
$$
L(x, y,\lambda)=f(x,y)-\lambda g(x,y)\\
$$
​	By solving the system of equation, we would get the optimized x, and y, then we can get the maximized value for $f(x)$.



#### Larange Multiplier in Dual SVM.

​	In our problem, we have our Larange function as below:
$$
L = \frac{1}{2}w^Tw+C\sum_{i=1}^{n}\epsilon_i+\sum_{i=1}^{n}\alpha_i(1-\epsilon_i- y_i(w^Tx_i+b))-\sum_{i=1}^{n}\mu_i\epsilon_i
$$
​	 Solving our Larange function, we need it fit **KKT condition**, which is 
$$
\begin{align*}
& \frac{\partial L}{\partial w}=w+\sum_{i=1}^{n}{\alpha_i}(-y_i)x_i=0\implies w=\sum_{i=1}^{n}{\alpha_iy_ix_i}\\
& \frac{\part L}{\part b}= \sum_{i=1}^{n}{\alpha_iy_i}=0 \\
& \frac{\part L}{\part \epsilon} =C-\alpha_i-\mu_i=0\implies \alpha_i=C-\mu_i\\
&\implies C\geq\alpha_i\geq0\ (Since\ \alpha_i\geq0,\ \mu_i\geq0)\\

\end{align*}
$$
​	We obtain three import conclusions by KKT condition, which are:
$$
\begin{align*}
& w=\sum_{i=1}^{n}{\alpha_iy_ix_i}\\
& \sum_{i=1}^{n}{\alpha_iy_i}=0\\
& C\geq\alpha_i\geq0
\end{align*}
$$
​	As we substitute conclusion we gained from KKT back into the Larange function, we will have:
$$
\begin{align*}
L&=\frac{1}{2}w^Tw+\sum_{i=1}^{n}\alpha_i(1-y_i(w^Tx_i+b))\\
&=\frac{1}{2}\sum_{i=1}^{n}{\alpha_iy_ix_i^T}\sum_{j=1}^{n}{\alpha_jy_jx_j}+\sum_{i=1}^{n}\alpha_i(1-y_i(\sum_{j=1}^{n}{\alpha_jy_jx_j^T}x_i+b))\\
&=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}+\sum_{i=1}^{n}\alpha_i-\sum_{i=1}^{n}\alpha_iy_i\sum_{j=1}^{n}\alpha_jy_jx_j^Tx_i-b\cdot \sum_{i=1}^{n}\alpha_iy_i\\
&=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}+\sum_{i=1}^{n}{\alpha_i}
\end{align*}
$$
​	We notice that the Lagrangian function is linear in α, we cannot set the gradient with respect to $\alpha $ to zero. We obtain the following dual optimization problem.
$$
\max_{\alpha}L(\alpha)=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}+\sum_{i=1}^{n}{\alpha_i}\\
\begin{align*}
s.t:\ \sum_{i=1}^{n}{\alpha_iy_i}=0\\
C\geq\alpha_i\geq0
\end{align*}
$$
​	**Till now, we have already constructed all the mathmatic model for the Soft Margain Dual SVM problem.**



#### Implementation of Soft Margain Dual SVM with CVXOPT.

​	Now we start to implement Soft Dual SVM with CVXOPT. We will use the the function "cvxopt_solvers.qp" in CVXOPT to help us solves the Lagrange function. "qp" solve for a specific function as below:
$$
minimze\ \frac{1}{2}x^TPx+q^Tx\\
\begin{align*}
subject\ to\ Gx\leq h\\
Ax=b
\end{align*}
$$
​	We now rewrite our function in order to fit the input for "cvxopt_solvers.qp".
$$
\begin{align*}
L(\alpha)&=
-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}+\sum_{i=1}^{n}{\alpha_i}\\

&=\frac{1}{2}\alpha^T(yX)^T(yX)\alpha+
\begin{bmatrix}
1,1,...,1
\end{bmatrix}
\alpha\\
\end{align*}\\
$$

$$
\begin{align*}
respect\ to : & \begin{bmatrix}
1,0,...,0,0\\
0,1,\ddots,0,0\\
0,0,\ddots,1,0\\
0,0,...,0,1
\end{bmatrix}
\begin{bmatrix}
\alpha_1\\
\alpha_2\\
\vdots\\
\alpha_n
\end{bmatrix}\geq
\begin{bmatrix}
0\\0\\\vdots\\0
\end{bmatrix}\\
and &
\begin{bmatrix}
-1,0,...,0,0\\
0,-1,\ddots,0,0\\
0,0,\ddots,-1,0\\
0,0,...,0,-1
\end{bmatrix}
\begin{bmatrix}
\alpha_1\\
\alpha_2\\
\vdots\\
\alpha_n
\end{bmatrix}\geq
\begin{bmatrix}
C\\
C\\
\vdots\\
C
\end{bmatrix}
\end{align*}
$$

```python
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def LinearSVM_Dual(X, y, C):
    n, p = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X

    P = cvxopt_matrix(X_dash.dot(X_dash.T))
    q = cvxopt_matrix(-np.ones((n, 1)))
    G = cvxopt_matrix(np.vstack((-np.diag(np.ones(n)), np.identity(n))))
    h = cvxopt_matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)

    alphas = np.array(sol['x'])
    sol_time = time.time() - start_time
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > 1e-4).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)
    b = b[0]

    # Display results
    x = np.linspace(-1, 5, 20)
    plt.plot(x, (-b - (w[0] * x)) / w[1], 'm')
    plt.show()
    return w, b
```

##### 	Till here, we have already discussed the _Hard Margain SVM_, _Soft Margain SVM Primal(L1 regularzation)_, and _Soft Margain SVM Dual_. Now we are going to discuss about the SVM L2 regularzation problem.



### SVM with L2 regularzation

​	After discussing about the SVM under Soft Margain, or L1 regularzation. We now consider if we can have a SVM model under L2 regularzation. The answer is definetly yes, and now we are going to build a L2 regularzation Primal and Dual model.

​	Firstly, we need to construct model, and turn the SVM into a convex optimization problem. Before all, lets recall the difference between L1 regularzation and L2 regularzation.
$$
\begin{align*}
L1\ reg:\min \frac{1}{2}||w||^2+C\sum_{i=1}^{n}{\epsilon_i}\ such\ that:\ \epsilon_i\geq0\\ 
L2\ reg:\min \frac{1}{2}||w||^2+\frac{C}{2}\sum_{i=1}^{n}{\epsilon_i^2}\ such\ that:\ \epsilon_i\ngeq0
\end{align*}
$$
​	As we see, in $l2\ norm\ SVM$, we are minimizing the $\sum_{i=1}^{n}{\epsilon_i^2}$, instead of $\sum_{i=1}^{n}{\epsilon_i}$. This will leads to a different conclusion in KKT condistion, and thus a different Larangian function. 



​	**_Important Note_:** We see we remove the constraint $\epsilon_i \geq 0$ in l2 reg, this is because consider a potential solution to the problem with some$\epsilon_i<0$. Then the constraint would also be fit as $\epsilon_i=0$, and the objective function will be lower in this situation, proving that this could not be an optimial solution. 



​	We now compute the Larangian function for our L2 norm SVM. 
$$
L(\alpha) =\frac{1}{2}||w||^2+\frac{C}{2}\sum_{i=1}^{n}{\epsilon_i^2}-\sum_{i=1}^{n}{\alpha_i(y_i(w^Tx+b)-1+\epsilon_i)}
$$
​	To get the optimal solution the following KKT conditions must be satisfied:
$$
\begin{align*}
& \frac{\part L}{\part w} =w-\sum_{i=1}^{n}\alpha_iy_ix_i=0;\\
& \frac{\part L}{\part b} =\sum_{i=1}^{n}y_i\alpha_i=0;\\
& \frac{\part L}{\part \epsilon}=C\cdot \epsilon_i-\alpha_i=0;\\
& \alpha_i(y_i(w^Tx_i+b)-1+\epsilon_i)=0
\end{align*}\\
$$
​	And our objective function will be:
$$
\begin{align*}
\max_\alpha\ & L(\alpha)=\sum_{i=1}^{n}\alpha_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}-\frac{1}{2}\sum_{i=1}^{n}\frac{\alpha_i^2}{C}\\
s.t.\ & \alpha\geq 0\\
& \sum_{i=1}^{n}{\alpha_iy_i=0}
\end{align*}
$$
​	Back to out qp solver, we need to rewrite our larangian function:
$$
\begin{align*}
L(\alpha)&=
\sum_{i=1}^{n}\alpha_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}-\frac{1}{2}\sum_{i=1}^{n}\frac{\alpha_i^2}{C}\\

&=\frac{1}{2}\alpha^T(y^TyX^TX+\frac{1}{C}I)\alpha-
1^T\alpha\\
\end{align*}\\
$$
​	Now we can start to implement our L2 reg Soft SVM:

```python
def l2_norm_LinearSVM_Dual(X, y, C):
    zero_tol = 1e-4
    n, p = X.shape
    y = y.reshape(-1, 1) * 1.
    X_dash = y * X
    extra_term = 1/C * np.identity(n)

    P = cvxopt_matrix(X_dash.dot(X_dash.T) + extra_term)
    q = cvxopt_matrix(-np.ones((n, 1)))
    G = cvxopt_matrix(np.vstack((-np.diag(np.ones(n)))))
    h = cvxopt_matrix(np.hstack((np.zeros(n))))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)

    alphas = np.array(sol['x'])
    w = ((y * alphas).T @ X).reshape(-1, 1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > zero_tol).flatten()

    # Computing b
    b = y[S] - np.dot(X[S], w)
    b = b[0]

    # Display results
    x = np.linspace(-1, 5, 20)
    plt.plot(x, (-b - (w[0] * x)) / w[1], 'm')
    plt.show()
    return w, b
```



### Customized Kernel SVM

​	Till now, we finished all the discussion of Linear SVM, we can now seperate data linearly in 2 dimension, or higher dimension. Now we come into a question, how do we make a decision boundary for points that can not be linear seperated. To begin with, lets start with the simplest example, seperating the following datasets.

<img src="/img/1D_Kernel_SVM_question.png" alt= "hi">

​	In the graph above, we can easily see that green points and red points that is clustered, or we can considering a problem such that we need to find out the amount of dosage of a certain medicine, green points representing the correct dosage, while red points representing either the medicine is under dosage or over dosage. Our question is to find the boundary of correct amount of dosage. 



​	Intuitively, we may consider to divide the data set into two parts, with each parts containing all the points on one side of green points, and green points itself. Then, we can apply our Linear SVM algorithm we implemented before to draw two decision boundaries, one on each side, to seperate our data.



​	This sounds like a very good solution, but considering our data goes into 2 dimension, we now can not solve with the problem with the same idea. For example, in the 2D seperation problem below, we would need infintly many axis to help us make a linear SVM, ie x-axis, y-axis, y=$\frac{1}{2}$x, y=$-\frac{1}{2}$x...

<img src="/img/2D_Kernel_SVM_problem.png" alt="hi" style="zoom:36%;" />

​	However, as we may consider in 2D non-linear seperating problem as the idea which we found as not useful as above, we consider to make a projection of the graph into different lines, and try to make a decision boundary on those lines. Now, what if we are considering this problem in another way, like if we want to project all the points into a higher dimensional space, will the points can be linearly seperatable? 

​	The answer is Yes.

​	Now lets go back to the simpliest 1D case.  Considering we are using a function $f(x)=x^2$, projecting each points into a 2D space, then we could find out that we can now seperate all the projected data points linearly, by using Linear SVM we discussed before.

<img src="/img/1D_projection_Kernel.png" alt="hi" style="zoom:33%;" /> 



​	Intuitively, we now ask ourself, if $f(x)=x^2$ is the best function for our projection? Can it be some function like $f(x)=x^3$?  or$f(x)=logx$ ?, and how do we find the best projection function?



​	Now we met the core of **Non-linear SVM**, or **Customized Kernel SVM**. 



### The Math behind Customized Kernel SVM

​	Recall that in Dual Soft Margain L1 SVM problem, we are maxmizing the larangian function:
$$
\max_{\alpha}L(\alpha)=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_jx_i^Tx_j}+\sum_{i=1}^{n}{\alpha_i}\\
\begin{align*}
s.t:\ \sum_{i=1}^{n}{\alpha_iy_i}=0\\
C\geq\alpha_i\geq0
\end{align*}
$$
​	Now, in Kernel SVM, we are going to maximizing the Larangian function:
$$
\max_{\alpha}L(\alpha)=-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}{\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j)}+\sum_{i=1}^{n}{\alpha_i}\\
\begin{align*}
s.t:\ \sum_{i=1}^{n}{\alpha_iy_i}=0\\
C\geq\alpha_i\geq0
\end{align*}
$$
​	Where here we take $\phi (x)$ as our projection function, and inner product of $\phi(x_i)^T\phi(x_j)$ as our **Kernel function** $K(x_i, x_j)$.



### Kernel functions in SVM

​	As we may consider, what are the kernel functions in SVM, and here we will have some examples. 

​	Firstly, we will have a **polynomial kernel** function, defining as: $K(x^Ty+b)^d$. Then we have a **Radial basis function kernel** with $\sigma$ :$K(x,y)=tanh(kx^Ty+\theta)$,  and **Sigmod Kernel** with parameter $k, and\ \theta$ : $K(x, y)=tanh(kx^Ty+\theta)$. Most importantly, we have the **Gaussian Kernel**, or kwon as "RBF", is defined as:$K(x, y)= e^{-\gamma||x-y||^2}$



### Gaussian Kernel

​	Here we will focus discussing the **Gaussian Kernel**. We will discuss Gauassian Kernel in 3 parts: firstly, we will focus **why** we are choosing Gaussian Kernel to help us do non-linear seperation, and how it can reflect the original data into infinite dimensions? Second part we will focus how do **implement the Gassian Kernel**. The Third part, we will focus on how to **use the Gaussian Kerel in our SVM**, and **plot the decision boundary**, using sklearn.



***NOTE: If you are not learning this in your machine learning course, or you are not highly interested in how it works, you can skip part1 and part2, part 3 is enough for practical usage. Part 1 and Part 2 will focus on the Math reasoning behind the algorithm. 



##### Part 1. Why Gaussian Kernel perform the best in seperating non-linear data?

​	Recall in our polynomial kernel, $K(x,y)=(x^Ty+b)^d$.

​	Consider in the kernel problem, we let $b=0, d=1$, we would have $K(x, y)=x^1y^1$. Intuitionally, this is a mapping that map the points in 1D from itself to itself, without changing anything.

​	Now consider and if consider a kernel with $b=0,d=1$,  then we would have $K(x,y)=x^2y^2$. Intuitionally, this is a mapping that maps the points in 1D from itself to its squared coordinates. The following graph shows a simple mapping of two points on 1D.

<img src="/img/d=2, polynomial.png" alt="hi" style="zoom:36%;" />

​	We see that the distance between these two points increases when we increasing the value of d.

​	

​	Now we know the method to increase the distance between points, but we still considering the question in 1D. Consider adding two kernels together, such that $K(x,y)= (x^Ty+0)^1+(x^Ty+0)^2=x^1y^1+x^2y^2$ , we see the result of such kernel is equal to $K(x,y)=(a,a^2)\cdot(b, b^2)$. This is a $\R^1 \implies \R^2$kernel, where the first coordinate is the original coordinate, and the second coordinate is the $y$ coordinate. This will be the mapping we mentioned before, to help us seperate the data in higher dimension.



​	Since we see that with larger d, the distance between original data are increased, so if we can map our toy 1D data into a infinite nD space, we may can find a $(n-1)D$ hyperplane to help us separate the data. Then our kernel would be :
$$
K(x,y)= a^1b^1+a^2b^2+a^3b^3+\dots+a^\infin b^\infin=(a,a^2,a^3,\dots,a^\infin)\cdot(b,b^2,b^3,\dots,b^{\infin})
$$
​	Recall that our Gaussian kernel function is: $e^{-\gamma||x-y||^2}$, and the **Taylor expansion** of $e^x$ is:
$$
e^{x}=e^a+\frac{e^a}{1!}(x-a)+\frac{e^a}{2!}(x-a)^2+\frac{e^a}{3!}(x-a)^3+\dots+\frac{e^a}{\infin!}(x-a)^\infin
$$
​	Expand the series at a=0, we have:
$$
e^{x}=1+\frac{1}{1!}x+\frac{1}{2!}x^2+\frac{1}{3!}x^3+\dots+\frac{1}{\infin!}x^\infin
$$
​	We notice that we can rewrite this into:
$$
\begin{align*}
e^{ab}&=1+\frac{1}{1!}ab+\frac{1}{2!}a^2b^2+\frac{1}{3!}a^3b^3+\dots+\frac{1}{\infin!}a^\infin b^\infin\\
&=(1,\sqrt{\frac{1}{1!}}a,\sqrt{\frac{1}{2!}}a^2,\sqrt{\frac{1}{3!}}a^3,\dots,\sqrt{\frac{1}{\infin!}}a^\infin)\cdot(1,\sqrt{\frac{1}{1!}}b,\sqrt{\frac{1}{2!}}b^2,\sqrt{\frac{1}{3!}}b^3,\dots,\sqrt{\frac{1}{\infin!}}b^\infin)
\end{align*}
$$
​	If we add one more constant $s=\sqrt{e^{-\frac{1}{2}(a^2+b^2)}}$into each term in dot product, we have:
$$
\begin{align*}
e^{-\frac{1}{2}(a-b)^2}&= e^{-\frac{1}{2}(a^2+b^2)}\cdot e^{ab} \\
&= e^{-\frac{1}{2}(a^2+b^2)} \cdot (1,\sqrt{\frac{1}{1!}}a,\sqrt{\frac{1}{2!}}a^2,\sqrt{\frac{1}{3!}}a^3,\dots,\sqrt{\frac{1}{\infin!}}a^\infin)\cdot(1,\sqrt{\frac{1}{1!}}b,\sqrt{\frac{1}{2!}}b^2,\sqrt{\frac{1}{3!}}b^3,\dots,\sqrt{\frac{1}{\infin!}}b^\infin)\\
&= (s,\sqrt{\frac{1}{1!}}a s,\sqrt{\frac{1}{2!}}a^2 s,\sqrt{\frac{1}{3!}}a^3 s,\dots,\sqrt{\frac{1}{\infin!}}a^\infin s)\cdot(s,\sqrt{\frac{1}{1!}}b s,\sqrt{\frac{1}{2!}}b^2 s,\sqrt{\frac{1}{3!}}b^3 s,\dots,\sqrt{\frac{1}{\infin!}}b^\infin s)
\end{align*}
$$
​	Thus, we found that our Gaussian Kernel, or "RBF" is actually a mapping to $\R^\infin$, and that is the reason "RBF" is Math behind why Gaussian Kernel is good for seperate non-linear data.



#####  Part 2: How do we implement the Gaussian Kernel?

​	To implement the Kernel by hand, we will have:

```python
def gaussianKernel(X1, X2, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum(np.power((x1 - x2), 2)) / float(2*(sigma**2)))
    return gram_matrix
```



##### Using the Gaussian Kernel in SVM to seperate non-linear data.

```python
import pandas as pd

data2 = pd.read_csv('prob2data.csv', header=None).values
X = data2[:, 0:2]
y = data2[:, -1]

# Invert the data
for i in range(len(y)):
    if y[i] != 1:
        y[i] = -1

def customized_kernel_svm():
    # Note here we are using "rbf" as our customized gaussian kernel.
    # Also note that we would have gamma = 1/2 sigma as we condsidering a sigma in our Gaussian Kernel.
    clf = SVC(kernel="rbf", gamma=20, C=1)
    X, y = X_train, y_train
    clf.fit(X, y)
	
    x1s = np.linspace(min(X[:, 0]), max(X[:, 0]), 600)
    x2s = np.linspace(min(X[:, 1]), max(X[:, 1]), 600)
    points = np.array([[x1, x2] for x1 in x1s for x2 in x2s])

    # Compute decision function for each point, keep those which are close to the boundary
    dist_bias = clf.decision_function(points)
    bounds_bias = np.array([pt for pt, dist in zip(points, dist_bias) if abs(dist) < 0.05])

    # Visualize the decision boundary
    plt.scatter(X[:, 0], X[:, 1], color=["r" if y_point == -1 else "b" for y_point in y], label="data")
    plt.scatter(bounds_bias[:, 0], bounds_bias[:, 1], color="g", s=0.5, label="decision boundary")
    plt.show()
```



​	As a result, you might expect the graph as following for the dataset in repo.

<img src="/img/non-linear_SVM_sigma_0.1.png" alt="hi" style="zoom:36%;" />



​	Congratulations my friend, you have learnt all the Math and implementation of different kinds of SVM. Hope this helps you and have a good time using SVM in your own project! Cheers!











