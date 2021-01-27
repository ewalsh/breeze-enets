package ai.economicdatasciences.enets.cv

import com.github.timsetsfire.enets.GlmNet
import com.github.timsetsfire.enets.RlmNet
import com.github.timsetsfire.enets.utils.ModelFit
import breeze.linalg.{DenseMatrix, DenseVector, Axis, sum, *}

case class CvModelFit(b0: Double, b: DenseVector[Double], lambda: Double,
  alpha: Double, avgEval: Double, sdEval: Double, tailEval: Double)

/** Estimate Generalized Linear Model with Elastic Net Regularization
  * using methods decribed in Regularization Paths for Generalized
  * linear model via Coordinate Descent
  * and Elements of Statistical Learning
  * @constructor create a RlmNet model
  * @param feature DenseMatrix of doubles containing features
  * @param target DenseVector of doubles containing the target
  * @param nFolds Int for number of folds to cross validate
  * @param offset DenseVector of doubles containing the offset
  * @param family for the model, values include gaussian, poisson,
  * negbin, binomial.  Default is Gaussian
  * @param link for the expected value of the target.  Doesn't really
  * need to be included because it is set depending on the family choosen
  * @param alpha mixing parameter for L1 and L2 Regularization
  * @param tolerance for coordinate descent
  * @param standardizeFeatures boolean value, whether to standardize
  * the feature matrix to mean 0 and unit variance, default is true
  * @param standardizeTarget boolean value, whether to standardize
  * the target vector to mean 0 and unit variance, default is false
  * @param intercept boolean value, whether or not to include a constant,
  * default is true
  */
class CvGlmNet(
               features: DenseMatrix[Double],
               target: DenseVector[Double],
               nFolds: Int,
               offset: DenseVector[Double] = DenseVector(1d),
               family: String = "gaussian",
               link: String = "identity",
               lambdaSeq: DenseVector[Double] = DenseVector.zeros[Double](100),
               alpha: Double = 1d,
               tolerance: Double = 1e-7,
               standardizeFeatures: Boolean = true,
               standardizeTarget: Boolean = false,
               intercept: Boolean = true
             ) {
   /** fit the elastic net via cross validation
   */
   def fit(evaluator: (DenseVector[Double], DenseVector[Double]) => Double): CvModelFit = {
     val foldSize = features.rows / nFolds.toDouble
     //segment dataset
     val partitions = (0 to features.rows - 1).grouped(math.ceil(foldSize).toInt).toArray
     val ptSet = (0 to features.rows - 1).toSet
     //compute test error for each fold
     // val xValError = partitions.foldRight(scala.Vector.empty[ModelFit])((c, acc) => {
     val mfArr = partitions.map(part => {
       //training data points are all data points not in validation set.
       val trainIdx = ptSet.diff(part.toSet)
       val testIdx = part
       //training data
       val trainX = features(trainIdx.toIndexedSeq, ::).toDenseMatrix
       val trainY = target(trainIdx.toIndexedSeq).toDenseVector
       //test data
       val testX = features(testIdx.toIndexedSeq, ::).toDenseMatrix
       val testY = target(testIdx.toIndexedSeq).toDenseVector
       // create glmnet model
       val enet = new GlmNet(trainX, trainY, offset, family, link, lambdaSeq, alpha,
         tolerance, standardizeFeatures, standardizeTarget, intercept)
       // fit model
       enet.fit
       // extract lambda sequence
       val lbdSeq = enet.getLambdaSeq
       // get evaluation for each lambda
       val allFits = (0 to (lbdSeq.length - 1)).map(lbdId => {
         val eval = evaluator((testX * enet.b(::,lbdId)) + enet.b0(lbdId), testY)
         ModelFit(enet.b0(lbdId), enet.b(::,lbdId), lbdSeq(lbdId), alpha, eval)
       })
       // return an array of all fits for later evaluation
       allFits.toArray
     })
     // get error
     val errArr = mfArr.map(_.map(_.eval))
     val errDM = DenseMatrix(errArr:_*)
     // identify best average lambda
     val avgLdaErr = breeze.stats.mean(errDM(::, *))
     val sdLdaErr = breeze.stats.stddev(errDM(::, *))
     val radjLdaErr = (avgLdaErr + sdLdaErr).t
     val minRAdj = breeze.linalg.min(radjLdaErr)
     val bestId = radjLdaErr.toArray.zipWithIndex.filter(_._1 == minRAdj)(0)._2
     // build model on full data using best lambda
     val bestFitArr = mfArr.map(arr => arr(bestId))
     val avgBestIcept = breeze.stats.mean(DenseVector(bestFitArr.map(_.b0)))
     val avgBestBetas = breeze.stats.mean(DenseMatrix(bestFitArr.map(_.b):_*), Axis._0).t
     val bestLda = bestFitArr(0).lambda
     // worst evaluation
     val evalDV = DenseVector(bestFitArr.map(_.eval))
     val avgEval = breeze.stats.mean(evalDV)
     val sdEval = breeze.stats.stddev(evalDV)
     val worstEval = breeze.linalg.max(evalDV)

     CvModelFit(avgBestIcept, avgBestBetas, bestLda, alpha,
      avgEval, sdEval, worstEval)
   }
}
