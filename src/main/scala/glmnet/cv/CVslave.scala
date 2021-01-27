package ai.economicdatasciences.enets.cv

import java.util.UUID

import ai.economicdatasciences.enets.cv.AkkaGlmNet.AlphaQueryInput
import ai.economicdatasciences.enets.cv.CvGlmNet
import ai.economicdatasciences.enets.cv.CvModelFit
import akka.actor.{Actor, ActorLogging}
import breeze.linalg.{DenseVector, DenseMatrix}

// import scala.collection.mutable.MutableList

object AkkaGlmSlave {
  case class AkkaModelFit(id: Int, alpha: Double, mod: CvModelFit)
}

class AkkaGlmSlave(id: Int,
                   features: DenseMatrix[Double],
                   target: DenseVector[Double],
                   nFolds: Int,
                   evaluator: (DenseVector[Double], DenseVector[Double]) => Double,
                   offset: DenseVector[Double],
                   family: String,
                   link: String,
                   lambdaSeq: DenseVector[Double],
                   tolerance: Double,
                   standardizeFeatures: Boolean,
                   standardizeTarget: Boolean,
                   intercept: Boolean
) extends Actor with ActorLogging {

  import AkkaGlmSlave._

  // val slaveData = inputPartition

  def receive = {
    case AlphaQueryInput(alpha) => {
      // val outList = MutableList[CvModelFit]()
      // for(a <- alpha){
      //   log.info(s"slave running with alpha = ${a}")
      //   // run cv glmnet
      //   val enet = new CvGlmNet(features, target, nFolds, offset, family,
      //     link, lambdaSeq, a, tolerance, standardizeFeatures,
      //     standardizeTarget, intercept)
      //
      //   // fit the model with cv
      //   val mod = enet.fit(evaluator)
      //   outList += mod
      //
      //   log.info(s"slave with alpha = ${a} finished elastic net")
      // }
      val a = alpha(id)
      log.info(s"slave running with alpha = ${a}")
      // run cv glmnet
      val enet = new CvGlmNet(features, target, nFolds, offset, family,
        link, lambdaSeq, a, tolerance, standardizeFeatures,
        standardizeTarget, intercept)

      // fit the model with cv
      val mod = enet.fit(evaluator)
      // outList += mod

      log.info(s"slave with alpha = ${a} finished elastic net")

      // send back best in group
      // val minEval = breeze.linalg.min(DenseVector(outList.map(_.avgEval).toArray))
      // val bestInGroup = outList.filter(_.avgEval == minEval)(0)

      sender() ! AkkaModelFit(id, a, mod)

    }
  }
}
