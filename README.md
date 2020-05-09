## Support Vector Machine with L1, L2 regularzation, and Customized Kernel

A quick glimsp and tutorial of implementing SVM. 



### Why we need SVM? What kind of problem does SVM solves?

​	In data science and machine learning, we always encounter a problem such that we need to find a threshold for spliting the data. For example, suppose we have collected data from 14 different people, 7 obesity, 7 normal, and we plot their weights as shown below.<img src="/img/SVM_Split_data_1d.png" alt="hi" style="zoom:50%;" />

​	Our goal is to **find a threshold, or draw a line**, as a decision boundary, to help us distinguish two different cluster of data. For instance, if the orange line is the threshold, then we would classify the black point as obesed.

​	As the threshold can be placed anywhere between the green points and orange points, our goal not is to see if we can **optimize threshold**.

​	**Now we reach the core of SVM.**



## Solving the simplest problem, Hard Margain SVM

​	Now, suppose our data is in the nicest scenario, that the data can be seperated into two distinct sets, with no overlaping. We can apply Hard Margain SVM.

<img src="/img/Figure_1.png" alt="alt text" style="zoom:36%;" />

​	Just like shown in the Plot, we see that blue points and green points are clustered and without any overlapping. 

​	Now we are wondering **how do we sperate it**? 

​	Suppose we are drawing a line as our decision boundary L:
$$
L: w^tx+b = 0
$$
​	and we also find two lines that is parrallel to the decision boundary L, L1 and L2 such that:
$$
L1:w^tx+b=1\\
L2:w^tx+b=-1\\
$$
​	We want L1 to pass through at least one blue points, and L2 to pass through at least one green points. Besides, we want no point between our L1 and L2.

<img src="/img/Figure_2.png" alt="alt text" style="zoom:36%;" />

​	Now suppose we have another set of parameters of w and b, we will gain a new decision boundary as below

<img src="/img/Figure_3.png" alt="alt text" style="zoom:36%;" />

​	We see that when we change to another decision boundary, like showed in second plot, the distance between L1 and L2 are decreased. 

​	Our question now specify to **find a set of L1, L2 that make the distance between L1 L2 the greatest.**

​	**Note that our goal is equvalent to find a set of L and L1, such that the distance between L and L1 is the largest.**

​	This turns to:
$$
\begin{align*}
\max{|L\ L1|}&=\max{\frac{|w^Tx+b|}{||w||}}\\
&=max\frac{1}{||w||}\\\\
&=min{||w||}
\end{align*}
$$
​	Besides, we have constraint such that 
$$
w^Tx_i+b\geq+1, when\ y_i=+1\\
w^Tx_i+b\leq-1, when\ y_i=-1\\
$$
​	To simplify ourconstraint, we can get as below: 
$$
y_i(w^Tx_i)\geq1
$$
​	Note that it is a little bit tricky, but easy to see when you try it.



​	**Since we change the problem into a pure mathmatic optimization problem, now we can start our implementation of SVM using CVXPY.**





## Implementation of Hard Margain SVM using CVXPY

#### 	

#### 	Defination of data

​		Our data has two inputs, X and y. X is a np.ndarray of shape (n,2), which contains all the points' coodinates; y is a list of length n, which contains the type of the corresponding coordinates in X, 1 for type A, and -1 for type B.



​		Now we create our toy data for Hard Margain SVM.

```python
import numpy as np
np.random.seed(5)

# First we get two different cluster of points, 40 points each, centered as (3,3) and (-3,-3)
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

# Then we join the data into the designated format.
X = join_data(x1, x2)
y = [1] * len(x1[0]) + [-1] * len(x2[0])
```



#### Implmentation of Hard Margain SVM

​	Recall that we turn the Hard Margain SVM into a math optimization problem, such that
$$
min{||w||},\ with:\ y_i(w^Tx_i)\geq1
$$
​	With CVXPY, the problem is showed as below.

```python
import cvxpy as cp
import matplotlib.pylab as plt

D = X.shape[1]
N = X.shape[0]
W = cp.Variable(D)
b = cp.Variable()

# Calculate the loss, and constraint of the problem.
loss = cp.sum_squares(W)
constraint = [y[i]*(X[i]*W+b) >= 1 for i in range(len(y))]
prob = cp.Problem(cp.Minimize(loss), constraint)
prob.solve()

# Get the value of w and b.
w = W.value
b = b.value

# Plot the decision boundary.
x = np.linspace(-4, 4, 20)
plt.plot(x, (-b - (w[0] * x)) / w[1], 'r', label="decision boundary for hard SVM")
plt.legend()
plt.show()
```

​	As a output, you would expect the graph as below.

<img src="/img/hard_margain_SVM_toy_data.png" alt="alt text" style="zoom:36%;" />



#### Problem of Hard Margain SVM: What if the data is not so nicely clustered? or what if with some mineral overlapping?

​	In Hard Margain SVM, we are making a decision boundary that correctly classify every point. This seems like classify the data very precisely, but **Hard Margain SVM is very sensitve to outliers**. For example, if we add one outlier into our data, and we run Hard Margain SVM again, the decision boundary would be dramaticly influenced by the outlier, and be not accurate anymore.

<img src="/Users/jinhanmei/Desktop/TODO 2020 APRIL/JinhanM.github.io/img/Hard_margain_with_outlier.png" alt="alt text" style="zoom:36%;" />



#### A solution: Soft Margain SVM Primal

​	We see this plot perfectly demonstrated how outlier influenced the decision boundary. Now we are considering how can we lower the influence of the outlier. **A solution is that we allow some points to be missclassified**. We now remove the contraint that make sure every point is correctly classified, and now maximizing the distance between two support vector and penalize on the points that is missclassified with a coeffecient C. When C is large, the algorithm penalize more on points that is missclassified, allowing more points to be correctly classified, when C is small, the algorithm penalize less on the points that is missclassified, making potentially more points to be missclassified. **This method is what we called Primal Soft Margain SVM, or L1 regularized SVM.**

​	Mathmatically, we turned this problem into optimization question as following:
$$
Minimize\ \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\epsilon_i\\
subject\ to:\ y_i(w^Tx_i+b)\geq1-\epsilon_i,\ where\ \epsilon_i\geq0.
$$


#### A small discussion on loss function

​	Note here we only want to punish the points that is misclassified, so we defining the loss as following:
$$
\epsilon_i=\max\{0,1-y_i(w^Tx_i+b)\}
$$
​	Since the correctly classified point has:
$$
y_i(w^Tx_i+b)\geq0
$$
​	Thus, this function only punishes points that is not correctly classified, and we call this funcion **"hinge loss"**, as its plots looks like a hinge.

<img src="/img/hinge_loss.png" alt="alt text" style="zoom:36%;" />



#### Implementation of Primal Soft Margain SVM.

```python
D = X.shape[1]
N = X.shape[0]
W = cp.Variable(D)
b = cp.Variable()

# We use C = 1/4 here.
C = 1/4
loss = (0.5 * cp.sum_squares(W) + C * cp.sum(cp.pos(1 - cp.multiply(y, X * W + b))))
# Note that we change the loss function and remove the constraints here.
prob = cp.Problem(cp.Minimize(loss))
prob.solve()

# Get the value of w and b.
w = W.value
b = b.value

# Plot the decision boundary.
x = np.linspace(-4, 4, 20)
plt.plot(x, (-b - ((w[0]) * x)) / w[1], 'r', label="decision boundary for Soft Margain SVM")
plt.legend()
plt.show()
```

<img src="/img/decision_boundary_of_soft_svm_c=0.25.png" alt="alt text" style="zoom:36%;" />

​	In plots above, we see that the decision boundary is pretty close to the decision boundary is much more accurate than the Hard Margain SVM perfoms. We also notice that this decision boundary is pretty close to the Hard Margain SVM without outlier, which implies that with this method, we can now dramatically reduce the impact of outliers damaging the accuracy of the decision boundary.



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

![alt text](/img/1D_Kernel_SVM_question.png)

​	In the graph above, we can easily see that green points and red points that is clustered, or we can considering a problem such that we need to find out the amount of dosage of a certain medicine, green points representing the correct dosage, while red points representing either the medicine is under dosage or over dosage. Our question is to find the boundary of correct amount of dosage. 



​	Intuitively, we may consider to divide the data set into two parts, with each parts containing all the points on one side of green points, and green points itself. Then, we can apply our Linear SVM algorithm we implemented before to draw two decision boundaries, one on each side, to seperate our data.



​	This sounds like a very good solution, but considering our data goes into 2 dimension, we now can not solve with the problem with the same idea. For example, in the 2D seperation problem below, we would need infintly many axis to help us make a linear SVM, ie x-axis, y-axis, y=$\frac{1}{2}$x, y=$-\frac{1}{2}$x...

<img src="/img/2D_Kernel_SVM_problem.png" alt="alt text" style="zoom:36%;" />

​	However, as we may consider in 2D non-linear seperating problem as the idea which we found as not useful as above, we consider to make a projection of the graph into different lines, and try to make a decision boundary on those lines. Now, what if we are considering this problem in another way, like if we want to project all the points into a higher dimensional space, will the points can be linearly seperatable? 

​	The answer is Yes.

​	Now lets go back to the simpliest 1D case.  Considering we are using a function $f(x)=x^2$, projecting each points into a 2D space, then we could find out that we can now seperate all the projected data points linearly, by using Linear SVM we discussed before.

<img src="/img/1D_projection_Kernel.png" alt="alt text" style="zoom:33%;" /> 



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

<img src="/img/d=2, polynomial.png" alt="alt text" style="zoom:36%;" />

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

<img src="/img/non-linear_SVM_sigma_0.1.png" alt="alt text" style="zoom:36%;" />



​	Congratulations my friend, you have learnt all the Math and implementation of different kinds of SVM. Hope this helps you and have a good time using SVM in your own project! Cheers!











