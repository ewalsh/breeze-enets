# Notes for fork
This fork builds on the great work done by @timsetsfire. The changes I made were to:
1. Perform a simple update to scala 2.12 and breeze 1.0.

2. I personally prefer Apache Zeppelin to Jupyter and so I have added a Docker image to make this easier to run/test.

You can get this working by running `docker-compose up` in the base directory
Then go to http://localhost:8080

3. Prediction and evaluation methods.

4. Cross-validation for model selection

5. A powerful part of this framework is to search over both lambda and alpha. I have added an
Akka distributed implementation to perform cross vaildation and selection of alpha.
  * You can see an implementation of this within the Zeppelin Notebook.

6. Added the Higham (2002) algorithm for non positive definite matrices.
  * The original project makes a positive definite assumption, but data
  is rarely this well structured in the wild.

~~~
import java.io.File
import breeze.linalg.csvread
import com.github.timsetsfire.enets.utils.mseEval
import ai.economicdatasciences.enets.cv.RunAkka._
import breeze.linalg.{DenseVector, DenseMatrix}

val x = csvread(new File("./conf/data/bostonX.csv"))
val y = csvread(new File("./conf/data/bostonY.csv")).toDenseVector
val nfld = x.rows / 3
setVariables(x, y, nfld, 0.1, mseEval, tolerance = 1e-3)
fitAkka
~~~

# ENETS (Elastic Nets in Scala)

Simple framework for running generalized linear models or robust linear models with elastic net regularization in scala.  Optimization is completed via coordinate descent as laid out in [Regularization Paths for Generalized Linear Models via Coordinate Descent](http://web.stanford.edu/~hastie/Papers/glmnet.pdf)

## Families

Via GlmNet class you can run the following generalized linear models with elastic net regularization:
* Gaussian with identity link
* Poisson with log link
* Binomial with logit link
* Negative Binomial with log link

## Robust Norms

Via RlmNet class, you can run a robust linear model with elastic net regularization with one of the following robust norms.  

Availabe robust norms include
* Least Squares - default norm
* Huber T
* Ramsey E
* Tukey Biweight
* Cauchy
* Trimmed Mean
* Approximate Huber (smooth)
* L1

The robust norms are based on the implementation in statsmodels for Python.  

## Requirements

[Breeze](https://github.com/scalanlp/breeze).

[SBT](www.scala-sbt.org)

## Usage

~~~
import breeze.linalg.{DenseVector,DenseMatrix}
import breeze.stats.distributions.{Gaussian, Poisson,Binomial}
import com.github.timsetsfire.enets._
import com.github.timsetsfire.enets.robust.norms._
val x = DenseMatrix.rand(100,50,Gaussian(0,4))
val y = DenseVector.rand(100,Gaussian(0,4))
val g = DenseVector.rand(100,Binomial(1,0.3)).map{_.toDouble}
// logistic regression
val lr = new GlmNet(x,g,family="binomial",link="logit")
lr.fit
lr.plotCoordinatePath  // Don't use this when x.cols >>  It takes forever!
// linear regression
val linreg = new GlmNet(x,y) // or new RlmNet(x,y)
linreg.fit
// huber regression
val hr = new RlmNet(x,y,rnorm=HuberT())
hr.fit

~~~
If you want to be able to use this in jupyter-scala notebooks, change the scala version in build.sbt to 2.11.x and run

`sbt package`

Then from jupyter notebook

`classpath.addPath("$PATH/enets/target/scala-2.11/enets_2.11-1.0.jar")`

and

`classpath.add("org.scalanlp" %% "breeze" % "0.13.2",
    "org.scalanlp" %% "breeze-natives" % "0.13.2",
    "org.scalanlp" %% "breeze-viz" % "0.13.2")`

See the usage notebook in the notebooks folder.
