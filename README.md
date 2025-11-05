
**patchReg: Patchwise Regression**

**An R package to increase predictive performance of a given fit**

*Summary*

The method coded in this package quickly and conveniently converts a
parametric model, say linear or logistic, into a "machine learning,"
i.e. nonparametric one.

*Introduction*

We all know that linear models -- in this case meaning linear in X, not
just in &beta; -- are just approximations. Sometimes they produce
acceptably accurate predictions, but in many cases one might seeks way
to enhance performance.

One possibility would be to add quadratic terms to the model. But if one
includes cross-product terms, e.g. X_1 X_2 in addition to X_1^2 and
X_2^2, the model size increases rapidly.

If p = 1, i.e. there is just one X variable, another possibility would
be piecewise linear regression, as shown [here](https://www.statology.org/piecewise-regression-in-r) and [here](https://www.r-bloggers.com/2023/12/unraveling-patterns-a-step-by-step-guide-to-piecewise-regression-in-r/).

But in the multivariate case, things are much less clear. It's easy to
partition the real line into intervals, and fit a separate linear model
within each interval, but how might one do this in more than one
dimension? (The above examples produce continuous fits, but we will not
require that here.)

Recently I came up with the idea of implementing partitioning via
k-means clustering. At first I thought my idea was new, but a literature
search showed it to be old. For instance, see [here](https://arxiv.org/abs/1211.1513) and [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8657940).
But as far as I know, nothing in this genre exists in CRAN etc., so I've
written it up [here](https://github.com/matloff/patchReg).

*Example*

The dataset **svcensus** in the [qeML
package](https://cran.r-project.org/package=qeML) is derived from the
2000 US census, with data on programmers and engineers..

``` r
> library(qeML)
> library(lattice)
> data(svcensus)
> head(svcensus)
       age     educ occ wageinc wkswrkd gender
1 50.30082 zzzOther 102   75000      52 female
2 41.10139 zzzOther 101   12300      20   male
3 24.67374 zzzOther 102   15400      52 female
4 50.19951 zzzOther 100       0      52   male
5 51.18112 zzzOther 100     160       1 female
6 57.70413 zzzOther 100       0       0   male
```

Say we are predicting Y = wageinc. Let's try partitioning the space into
2 *patches*, with linear regression:

``` r
> source('patchReg.R')
> args(patchReg)
function (XYdata, yName, numClust, regCall, savePreds = TRUE,
    regPredFtn = NULL, classPredFtn = NULL, holdout = floor(min(1000,
        0.1 * nrow(XYdata))))
NULL
> z <- patchReg(svcensus,'wageinc',2,"function(xy) lm(wageinc ~ .,xy)")
```

In that last argument, we are specifying the call for fitting to the
dataset **xy**, which will be one of the patches.

``` r
> z

testAcc: 23116.55

centers:

       age   educ.14    educ.16   occ.100   occ.101   occ.102    occ.106
1 39.03322 0.1771218 0.01992620 0.2516605 0.2793358 0.2937269 0.01808118
2 39.62099 0.2210012 0.03876679 0.2252747 0.2154457 0.3464591 0.02539683
     occ.140  wkswrkd gender.female
1 0.03431734 11.59004     0.2830258
2 0.04096459 50.68571     0.2375458

coefficients:

  (Intercept)           age       educ.14       educ.16       occ.100
  -3515.35646     123.84234    4552.93417    1801.45178   -3153.83111
      occ.101       occ.102       occ.106       occ.140       wkswrkd
  -1882.73335    2040.52972   -4665.84524    3096.84194    1093.86439
gender.female
     75.67962
  (Intercept)           age       educ.14       educ.16       occ.100
  -49832.2226      602.4936    15558.7274    22360.9963   -11540.8776
      occ.101       occ.102       occ.106       occ.140       wkswrkd
  -10492.4264     3133.0877    -9458.4017      927.9902     1895.8642
gender.female
  -10143.0795
```

We see from the display of k-means centers that our partitioning most
reflects differences in number of weeks worked, and gender, with some
differences in occupation as well.

We have two sets of estimated regression coefficients, one for each
cluster. Not surprisingly, they differ a lot from each other, an
indication that patchwise regression may be useful here.

To predict a new case, the code determines which cluster the new case
falls into, then uses the coefficients for that case to find the
predicted value.

By default there is a random holdout set extracted, thus forming
training and test sets. The k-means clustering is done on the training
set. The accuracy, Mean Absolute Prediction Error here, is calculated on
the test set.

We may also view the distribution of predicted values from the holdout
set, for each cluster.

![loc](DensPLot.png)


*Performance: Regression*

Let's use cross-validation whether patchwise regression yieldss
performance gains.

``` r
> replicMeans(500,"patchReg(svcensus,\"wageinc\",1,\"function(xy)
  lm(wageinc ~ .,data=xy)\")$testAcc")
[1] 25389.57
attr(,"stderr")
[1] 46.56158
> replicMeans(500,"patchReg(svcensus,\"wageinc\",2,\"function(xy)
  lm(wageinc ~ .,data=xy)\")$testAcc")
[1] 24840.45
attr(,"stderr")
[1] 48.21145
> replicMeans(500,"patchReg(svcensus,\"wageinc\",4,\"function(xy)
  lm(wageinc ~ .,data=xy)\")$testAcc")
[1] 24428.92
attr(,"stderr")
[1] 46.92762
replicMeans(500,"patchReg(svcensus,\"wageinc\",8,\"function(xy)
lm(wageinc ~ .,data=xy)\")$testAcc")
[1] 24435.79
attr(,"stderr")
```

Seems that either 4 or 8 clusters works best here. Here is a run with
the well-known Bike Rentals dataset:

``` r
> replicMeans(500,"patchReg(dy2,\"tot\",1,\"function(xy) lm(tot ~
> .,data=xy)\")$testAcc")
[1] 575.6359
attr(,"stderr")
[1] 2.825645
> replicMeans(500,"patchReg(dy2,\"tot\",2,\"function(xy) lm(tot ~
> .,data=xy)\")$testAcc")
[1] 541.2743
attr(,"stderr")
[1] 2.948977
There were 50 or more warnings (use warnings() to see the first 50)
> replicMeans(500,"patchReg(dy2,\"tot\",4,\"function(xy) lm(tot ~
> .,data=xy)\")$testAcc")
[1] 513.9532
attr(,"stderr")
[1] 4.177588
There were 50 or more warnings (use warnings() to see the first 50)
> replicMeans(500,"patchReg(dy2,\"tot\",8,\"function(xy) lm(tot ~
> .,data=xy)\")$testAcc")
[1] 548.9618
attr(,"stderr")
[1] 3.671386
```

Not surprisingly, patchwork regression works very well in the univariate
case. In the **svcensus** dataset, income as a nonmonotone relation wtih
age (it is increasing in one's 20s, declining after 50 or so).

``` r
> replicMeans(500,"patchReg(svcensus,\"wageinc\",1,\"function(xy) lm(wageinc ~ age ,data=xy)\")$testAcc")
[1] 31815.33
attr(,"stderr")
[1] 49.83686
> replicMeans(500,"patchReg(svcensus,\"wageinc\",2,\"function(xy) lm(wageinc ~ age ,data=xy)\")$testAcc")
[1] 28385.42
attr(,"stderr")
[1] 77.09606
Warning message:
Quick-TRANSfer stage steps exceeded maximum (= 954500)
> replicMeans(500,"patchReg(svcensus,\"wageinc\",4,\"function(xy) lm(wageinc ~ age ,data=xy)\")$testAcc")
[1] 26968.2
attr(,"stderr")
[1] 47.39117
> replicMeans(500,"patchReg(svcensus,\"wageinc\",8,\"function(xy) lm(wageinc ~ age ,data=xy)\")$testAcc")
[1] 26335.84
attr(,"stderr")
[1] 47.88535
```

*Performance: Classification*

Here is an example using the **Vowel** dataset in the **mlbench**
package. The goal is to predict one of six vawels sounds, based on audio
measurements. 

We use the **qeML** function **qeLogit**.  The role of the argument
**classPredFtn** is as follows. In some models, e.g. **lm**, calling
**predict** returns actual predictions, but in some other the returned
object contains the predicted value as one of several components. The
latter is the case here.

``` rc
> replicMeans(50,"patchReg(Vowel,\"Class\",1,\"function(xy) qeLogit(xy,'Class',holdout=NULL)\",classPredFtn=classPredFtn.qeML)$testAcc")
[1] 0.4038384
attr(,"stderr")
[1] 0.006404651
There were 50 or more warnings (use warnings() to see the first 50)
> replicMeans(50,"patchReg(Vowel,\"Class\",2,\"function(xy) qeLogit(xy,'Class',holdout=NULL)\",classPredFtn=classPredFtn.qeML)$testAcc")
[1] 0.2379798
attr(,"stderr")
[1] 0.006350256
There were 50 or more warnings (use warnings() to see the first 50)
> replicMeans(50,"patchReg(Vowel,\"Class\",4,\"function(xy) qeLogit(xy,'Class',holdout=NULL)\",classPredFtn=classPredFtn.qeML)$testAcc")
[1] 0.139798
attr(,"stderr")
[1] 0.005881784
There were 50 or more warnings (use warnings() to see the first 50)
> replicMeans(50,"patchReg(Vowel,\"Class\",8,\"function(xy) qeLogit(xy,'Class',holdout=NULL)\",classPredFtn=classPredFtn.qeML)$testAcc")
[1] 0.09272727
attr(,"stderr")
[1] 0.004928985
There were 50 or more warnings (use warnings() to see the first 50)
```

(The warnings were as typical for **glm**, ie fitted probabilities of 1 or 0.)

*Machine Learning methods*

NOTE: WE ARE USING THE DEFAULT VALUES HERE

``` r
> replicMeans(50,"patchReg(svcensus,\"wageinc\",1,\"function(xy) qeRFranger(xy,'wageinc',holdout=NULL)\")$testAcc")
[1] 24742.62
attr(,"stderr")
[1] 143.227
> replicMeans(50,"patchReg(svcensus,\"wageinc\",2,\"function(xy) qeRFranger(xy,'wageinc',holdout=NULL)\")$testAcc")
[1] 24651.46
attr(,"stderr")
[1] 157.5083
> replicMeans(50,"patchReg(svcensus,\"wageinc\",4,\"function(xy) qeRFranger(xy,'wageinc',holdout=NULL)\")$testAcc")
[1] 24770.97
attr(,"stderr")
[1] 130.703
> replicMeans(50,"patchReg(svcensus,\"wageinc\",8,\"function(xy) qeRFranger(xy,'wageinc',holdout=NULL)\")$testAcc")
[1] 24240.84
attr(,"stderr")
[1] 136.5613
```

Patchwork regression is also typically much faster than ML methods:

``` r
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",1,\"function(xy) qeRFranger(xy,'wageinc',holdout=NULL)\")$testAcc"))
   user  system elapsed
 82.327   4.668  64.682
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",1,\"function(xy) qeLin(xy,'wageinc',holdout=NULL)\")$testAcc"))
   user  system elapsed
  4.847   0.000   4.846
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",2,\"function(xy) qeLin(xy,'wageinc',holdout=NULL)\")$testAcc"))
   user  system elapsed
  5.033   0.004   5.050
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",4,\"function(xy) qeLin(xy,'wageinc',holdout=NULL)\")$testAcc"))
   user  system elapsed
  5.290   0.000   5.291
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",8,\"function(xy) qeLin(xy,'wageinc',holdout=NULL)\")$testAcc"))
   user  system elapsed
  5.239   0.000   5.244

   user  system elapsed
159.320   0.459  20.603
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",1,\"function(xy) qeXGBoost(xy,'wageinc',holdout=NULL)\")$testAcc"))

   user  system elapsed
166.377   0.920  21.598
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",2,\"function(xy) qeXGBoost(xy,'wageinc',holdout=NULL)\")$testAcc"))

   user  system elapsed
190.237   1.371  24.969
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",4,\"function(xy) qeXGBoost(xy,'wageinc',holdout=NULL)\")$testAcc"))

   user  system elapsed
220.673   2.277  29.042
> system.time(replicMeans(5,"patchReg(svcensus,\"wageinc\",8,\"function(xy) qeXGBoost(xy,'wageinc',holdout=NULL)\")$testAcc"))
```

