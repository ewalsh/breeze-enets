package ai.economicdatasciences.enets.cv

import ai.economicdatasciences.enets.cv.AkkaGlmNet
import ai.economicdatasciences.enets.cv.AkkaGlmNet.AlphaQueryInput
import akka.actor.{ActorSystem, Props, Actor}
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.timsetsfire.enets.utils.mseEval
// import ai.economicdatasciences.enets.cv.AkkaGlmSlave.AkkaModelFit
import ai.economicdatasciences.enets.cv.CvModelFit

// import java.io.File
// import breeze.linalg.csvread

case class AkkaInput(features: DenseMatrix[Double], target: DenseVector[Double],
  nFolds: Int, alphaStep: Double,
  evaluator: (DenseVector[Double], DenseVector[Double]) => Double,
  offset: DenseVector[Double] = DenseVector(1d), family: String,
  link: String, lambdaSeq: DenseVector[Double] = DenseVector.zeros[Double](100),
  tolerance: Double, standardizeFeatures: Boolean,
  standardizeTarget: Boolean, intercept: Boolean)

class RunAkkaGlm extends Actor {

  import RunAkka._
  val inputVars = inputs.get
  val alphaStep = inputVars.alphaStep
  val nflds = inputVars.nFolds
  val aStepInt = (alphaStep * 100.0).toInt
  val aSteps = (0 to 100 by aStepInt).toArray.map(_.toDouble / 100.0)
  val numPartitions = aSteps.length
  var outputModel: Option[CvModelFit] = None

  val akkaCVmaster = context.actorOf(
    Props(new AkkaGlmNet(
      inputVars.features, inputVars.target, nflds, numPartitions, mseEval,
      inputVars.offset, inputVars.family, inputVars.link, inputVars.lambdaSeq,
      inputVars.tolerance, inputVars.standardizeFeatures,
      inputVars.standardizeTarget, inputVars.intercept)
    )
  )

  akkaCVmaster ! AlphaQueryInput(aSteps)

  def receive = {
    case CvModelFit(b0, b, lambda, alpha, avgEval, sdEval, tailEval) => {
      println(s"output err = ${avgEval}")
      outputModel = Some(CvModelFit(b0, b, lambda, alpha, avgEval, sdEval, tailEval))
      updateFittedModel(outputModel)
    }
  }

  override def postStop(): Unit = {
    println("Actor stopped")
  }
}

object RunAkka {

  var inputs: Option[AkkaInput] = None
  var fittedModel: Option[CvModelFit] = None

  // inputs: Option[AkkaInput]
  def fitAkka(): Unit = {
    inputs match {
      case None => println("Set Input Variables")
      case _ => {
        val system = ActorSystem("Main")
        val ac = system.actorOf(Props[RunAkkaGlm])
      }
    }
  }

  def setVariables(features: DenseMatrix[Double], target: DenseVector[Double],
    nFolds: Int, alphaStep: Double,
    evaluator: (DenseVector[Double], DenseVector[Double]) => Double,
    offset: DenseVector[Double] = DenseVector(1d), family: String = "gaussian",
    link: String = "identity",
    lambdaSeq: DenseVector[Double] = DenseVector.zeros[Double](100),
    tolerance: Double = 1e-7, standardizeFeatures: Boolean = true,
    standardizeTarget: Boolean = false, intercept: Boolean = true): Unit = {

      inputs = Some(AkkaInput(features, target, nFolds, alphaStep, evaluator,
        offset, family, link, lambdaSeq, tolerance, standardizeFeatures,
        standardizeTarget, intercept))

      println("variables set")
    }

  def updateFittedModel(mod: Option[CvModelFit]): Unit = {
    fittedModel = mod
    println("fitted model updated")
  }

  def getModel(): CvModelFit = {
    fittedModel.get
  }

  def predictAkka(newInputs: DenseMatrix[Double]): Option[DenseVector[Double]] = {
    fittedModel match {
      case None => {
        println("Fit Model before prediction")
        None
      }
      case _ => {
        val tmp = fittedModel.get
        Some((newInputs * tmp.b) + tmp.b0)
      }
    }
  }

}
