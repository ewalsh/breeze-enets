package ai.economicdatasciences.enets.cv

// import java.util.UUID
import akka.actor.{Props, ActorRef, Actor, ActorLogging}
import breeze.linalg.{DenseVector, DenseMatrix}
import ai.economicdatasciences.enets.cv.CvModelFit
import ai.economicdatasciences.enets.cv.AkkaGlmSlave
import ai.economicdatasciences.enets.cv.AkkaGlmSlave.AkkaModelFit

import scala.collection.mutable.MutableList

object AkkaGlmNet {
  case class AlphaQueryInput(alpha: Array[Double])
}

class AkkaGlmNet(
                 features: DenseMatrix[Double],
                 target: DenseVector[Double],
                 nFolds: Int,
                 numPartitions: Int,
                 evaluator: (DenseVector[Double], DenseVector[Double]) => Double,
                 offset: DenseVector[Double] = DenseVector(1d),
                 family: String = "gaussian",
                 link: String = "identity",
                 lambdaSeq: DenseVector[Double] = DenseVector.zeros[Double](100),
                 tolerance: Double = 1e-7,
                 standardizeFeatures: Boolean = true,
                 standardizeTarget: Boolean = false,
                 intercept: Boolean = true
               ) extends Actor with ActorLogging {

    import AkkaGlmNet._

    var outputModel: Option[CvModelFit] = None

    log.info(s"Searching Over ${numPartitions} Alphas")

    // create actors to handle each partition
    val partitionActors: Array[ActorRef] = new Array[ActorRef](numPartitions)

    (0 to (numPartitions - 1)).foreach({ case (id) =>
      partitionActors(id) = context.actorOf(Props(new AkkaGlmSlave(
        id,
        features,
        target,
        nFolds,
        evaluator,
        offset,
        family,
        link,
        lambdaSeq,
        tolerance,
        standardizeFeatures,
        standardizeTarget,
        intercept
      )))
    })

    log.info("Slave actors created")

    var slavesNotFinished = numPartitions
    val collectedResults = MutableList[CvModelFit]()

    def receive = {
      case AlphaQueryInput(alpha) => {
        partitionActors.foreach(_ ! AlphaQueryInput(alpha))
        context.become(waitForSlaves)
      }
    }

    def waitForSlaves: Receive = {
      case AkkaModelFit(id, a, mod) => {
        val a = mod.alpha
        log.info(s"slave ${a} search results received by master")

        slavesNotFinished -= 1
        // val pctComplete = ((numPartitions - slavesNotFinished)/numPartitions) * 100.0

        // println(s"${pctComplete.toInt}%")

        log.info(s"${slavesNotFinished} workers still working...")
        collectedResults += mod

        if(slavesNotFinished == 0){
          log.info("All results computed")

          val minAvgEval = breeze.linalg.min(DenseVector(collectedResults.map(_.avgEval).toArray))
          val finalMod = collectedResults.filter(_.avgEval == minAvgEval)(0)

          log.info("The best parameters are:")
          log.info(s"alpha = ${finalMod.alpha}")
          log.info(s"lambda = ${finalMod.lambda}")
          log.info(s"with an average error = ${finalMod.avgEval}")

          outputModel = Some(finalMod)

          context.parent ! finalMod
          context.unbecome()
        }
      }
    }

    def results(): CvModelFit = {
      outputModel.get
    }

}
