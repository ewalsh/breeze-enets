package com.github.timsetsfire.enets.optimize

import breeze.linalg._
import breeze.numerics._
import com.github.timsetsfire.enets.utils._


class CoordinateDescent( cost: (Params, Double, Double) => Double,
                         x: DenseMatrix[Double],
                         y: DenseVector[Double],
                         w: DenseVector[Double],
                         s: DenseVector[Double],
                         weightFunction: (DenseVector[Double]) => DenseVector[Double]) {

  def optimize(alpha: Double, lambda: Double, params: Params, tolerance: Double = 1e-6, nz: Array[Int] = Array.empty) = {
    //println("optimizing")
    def descend(params: Params, ind: Int): Unit = {
      val b = params.b(ind)
      val wx = (x(::, ind) *:* w) *:* s
      val t = 1/ (x.rows.toDouble) *( wx.t * ( (y - g(x,params) + (params.b(ind)*x(::, ind)) )))
      params.b(ind) = softThresh(t, alpha * lambda) / ( 1/x.rows.toDouble * (wx.t*x(::, ind)) + (1 - alpha)*lambda )
      if( abs( b - params.b(ind)) < tolerance) Unit
      else descend(params, ind)
    }
    def interceptDescend(params: Params): Unit = {
      val b0 = params.b0(0)
      params.b0(0) = 1 / sum(w*:* s) * ((w*:* s).t *( ( y - x*params.b)))
      if( abs(b0 - params.b0(0)) < tolerance) Unit
      else interceptDescend(params)
    }

    def update(params: Params, tolerance: Double = 1e-6): Unit = {
      val j = cost(params, lambda, alpha)
      //val randind = DenseVector.rand(params.nparams, Binomial(1, scd)).toArray.zipWithIndex.filter{ tup => tup._1 == 1 & tup._2 > 0}.map{ _._2}
      if(params.intercept) interceptDescend(params)
      for( ind <- nz) {
        descend(params, ind)
      }

      if( abs(j - cost(params, lambda, alpha)) < tolerance) {
        Unit
      }
      else {
        val r = (y - g(x, params))
        w := weightFunction(r)
        update(params, tolerance)
      }
    }
    update(params, tolerance)
    // var iter = 0
    // while(iter < 100){
    //   update(params, tolerance)
    //   iter += 1
    //   println(iter)
    // }
  }
}
