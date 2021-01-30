package com.github.timsetsfire.enets

// ewalsh fork -- added try catch for pseudoinverse matrix which can fail
// -- added mse evaluator


package object utils {

  import breeze.linalg.{*, DenseMatrix, DenseVector, diag, pinv, inv}
  import breeze.numerics.{abs, signum, sqrt}
  import breeze.stats.{mean, variance}
  import scala.util.Try
  import ai.economicdatasciences.enets.utils.NearPD

  case class ModelFit(b0: Double, b: DenseVector[Double], lambda: Double,
    alpha: Double, eval: Double)

  // instead of adding in a vector of ones to represent the bias, use the following with the Params case class
  def g(x: DenseMatrix[Double], b: Params) = {
    (x*b.b).map{ _ + b.b0(0)}
  }

  case class Params(
                     b: DenseVector[Double],
                     b0: DenseVector[Double] = DenseVector(0d),
                     intercept: Boolean = false,
                     scale: DenseVector[Double] = DenseVector(1d)
                   )
  {
    val nfeatures = b.length
    val nzbv = DenseVector.zeros[Double](nfeatures) >:> 0d  // used to identify non zero values
  }

  def softThresh(b: Double, gamma: Double) = {
    val sig = signum(b);
    val f = abs(b) - gamma  ;
    val pos = ( abs(f) + f ) /2;
    sig * pos;
  }

  def tryPseudoInv(mat: DenseMatrix[Double]): Try[DenseMatrix[Double]] = Try(pinv(sqrt(mat))) recoverWith {
    case exception => Try { //Try(inv(sqrt(mat)))
      // ensure positive definite
      val nearPD = new NearPD(mat)
      val nearestPD = nearPD.generate
      nearestPD
    }
  }

  def stdizeMatrix(x: DenseMatrix[Double]) = {
    val n = x.rows.toDouble
    val mu = mean(x(::,*))
    val sig = (n-1)/n*(diag(variance(x(::,*))))
    val pinvOrInv = tryPseudoInv(sig).get
    val xs = (x(*,::).map{ _ - mu.t})*pinvOrInv //pinv(sqrt(sig))
    (xs, mu, sig)
  }

  def stdizeVector(x: DenseVector[Double]) = {
    val n = x.length.toDouble
    val sig = (n-1)/(n) * variance(x)
    val mu = mean(x)
    val xs = 1/sqrt(sig) * (x - mean(x))
    (xs, mu, sig)
  }

  val mseEval = (pred: DenseVector[Double], target: DenseVector[Double]) =>
    breeze.stats.mean((pred - target).map(x => scala.math.pow(x, 2)))
}
