## Support Vector Machine with L1, L2 regularzation, and Customized Kernel

A quick glimsp and tutorial of implementing SVM. 



### Why we need SVM? What kind of problem does SVM solves?

In data science and machine learning, we always encounter a problem such that we need to find a threshold for spliting the data. For example, suppose we have collected data from 14 different people, 7 obesity, 7 normal, and we plot their weights as shown below.

<img src="/Users/jinhanmei/Desktop/TODO 2020 APRIL/JinhanM.github.io/img/SVM_Split_data_1d.png" alt="alt text" style="zoom:50%;" />

Our goal is to **find a threshold, or draw a line**, as a decision boundary, to help us distinguish two different cluster of data. For instance, if the orange line is the threshold, then we would classify the black point as obesed.

As the threshold can be placed anywhere between the green points and orange points, our goal not is to see if we can **optimize threshold**.

**Now we reach the core of SVM.**



## Solving the simplest problem, Hard Margain SVM

Now, suppose our data is in the nicest scenario, that the data can be seperated into two distinct sets, with no overlaping. We can apply Hard Margain SVM.

<img src="/Users/jinhanmei/Desktop/TODO 2020 APRIL/JinhanM.github.io/img/Figure_1.png" alt="alt text" style="zoom:36%;" />

Just like shown in the Plot, we see that blue points and green points are clustered and without any overlapping. 

Now we are wondering **how do we sperate it**? 

Suppose we are drawing a line as our decision boundary L:
$$
L: w^tx+b = 0
$$
and we also find two lines that is parrallel to the decision boundary L, L1 and L2 such that:
$$
L1:w^tx+b=1\\
L2:w^tx+b=-1\\
$$
We want L1 to pass through at least one blue points, and L2 to pass through at least one green points. Besides, we want no point between our L1 and L2.

<img src="/Users/jinhanmei/Desktop/TODO 2020 APRIL/JinhanM.github.io/img/Figure_2.png" alt="alt text" style="zoom:36%;" />

now suppose we have another set of parameters of w and b, we will gain a new decision boundary as below

<img src="/Users/jinhanmei/Desktop/TODO 2020 APRIL/JinhanM.github.io/img/Figure_3.png" alt="alt text" style="zoom:36%;" />

we see that when we change to another decision boundary, like showed in second plot, the distance between the two support vector, 





it is easy to see that the distance between red line and blue line is much more smaller than the previous one. 









and we find two lines that is parallel to L such as L1= 









[Comparing t-test and Mann Whitney test for the means of Gamma](https://xavierbourretsicotte.github.io/t_test_Mann_Whitney.html)Thu 18 October 2018 — Xavier Bourret Sicotte

This notebook explores various simulations where we are testing for the difference in means of two independent gamma distributions, by sampling them and computing the means of each sample. We will compare two main test methods: the t-test and the Mann Whitney test.

Category: [Statistics](https://xavierbourretsicotte.github.io/category/statistics.html) Cover:

