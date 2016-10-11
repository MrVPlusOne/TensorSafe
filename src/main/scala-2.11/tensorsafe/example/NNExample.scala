package tensorsafe.example

import tensorsafe.Implicits._
import tensorsafe._

import scala.util.Random

trait DataNum extends VarDim
trait InputDim extends VarDim
trait LabelNum extends VarDim
trait HiddenNum extends VarDim

/**
 * An example from the programming assignment of Stanford CS224d: Deep learning for NLP,
 * originally written in python.
 */
object NNExample {

  def sigmoid(x: Double) = 1.0/(1+math.exp(-x))
  def sigmoid_grad(f: Double) = f * (1-f)

  /**
   * Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
   */
  def forward_backward_prop(data: Tensor[DataNum~InputDim], labels: Tensor[DataNum~LabelNum],
      w1: Tensor[InputDim~HiddenNum], b1: Tensor[HiddenNum],
      w2: Tensor[HiddenNum~LabelNum], b2: Tensor[LabelNum]) = {


    // forward propagation
    val hidden = (data * w1 + b1).mapAll(sigmoid)
    val prediction = (hidden * w2) + b2
    val cost = -(prediction.mapAll(math.log) *^ labels).sumAll

    // backward propagation
    val delta = prediction - labels
    val gradW2 = hidden.t matMul delta
    val gradb2 = delta.sum(TNumber.n1)

    val delta2 = delta * w2.t *^ hidden.mapAll(sigmoid_grad)
    val gradW1 = data.t * delta2
    val gradb1 = delta2.sum(TNumber.n1)

    (cost, gradW1, gradb1, gradW2, gradb2)
  }

  /**
   * Gradient check for a function f
   * @param f should be a function that takes a single argument and outputs the cost and its gradients
   * @param x is the point to check the gradient at
   */
  def gradcheck_naive[S,Idx](f: Tensor[S]=> (Double, Tensor[S]), x: Tensor[S])(implicit s2i: ShapeToIndex[S,Idx], i2v: IndexToVec[Idx]): Unit = {
    val (_, grad) = f(x)
    val h = 1e-4

    x.indices.foreach(i=>{
      x(i) = x(i) + h
      val (fxh, _) = f(x)
      x(i) = x(i) - 2*h
      val (fxnh, _) = f(x)
      x(i) = x(i) + h

      val numericGrad = (fxh-fxnh) / (2*h)

      // Compare gradients
      import math.{abs, max}
      val relativeDiff =  abs(numericGrad-grad(i)) / max(max(1, abs(numericGrad)), abs(grad(i)))

      if(relativeDiff > 1e-5){
        println("Gradient check failed.")
        println(s"First gradient error found at index $i")
        println(s"Your gradient: ${grad(i)} \t Numerical gradient: $numericGrad")
      }
    })

    println("Gradient check passed!")
  }



  trait D1 extends VarDim
  trait D2 extends VarDim
  trait D3 extends VarDim

  def sanity_check(): Unit = {
    def quad[S](x: Tensor[S])(implicit b: ShapeBroadcast[S,S,S], b2: ShapeBroadcast[S,UnitDim,S]) = ((x*^x).sumAll, x *^ scalar(2))

    val d1 = dim[D1](3)
    val d2 = dim[D2](4)
    val d3 = dim[D3](5)

    gradcheck_naive(quad[UnitDim], scalar(123.456))
    gradcheck_naive(quad[D1], (TensorBuilder > d1).randGaussian)
    gradcheck_naive(quad[D2~D3], (TensorBuilder > d2 ^ d3).randGaussian)
  }

  def neural_sanity_check(random: Random): Unit = {
    val dataDim = dim[DataNum](20)
    val inputDim = dim[InputDim](10)
    val hiddenDeim = dim[HiddenNum](5)
    val labelDim = dim[LabelNum](10)

    val data = (tb > dataDim ^ inputDim).randGaussian
    val labels = (tb > dataDim ^ labelDim).zeros

    for(i <- 0 until dataDim.value){
      labels(i~random.nextInt(labelDim.value)) = 1
    }

    val w1 = (tb > inputDim ^ hiddenDeim).randGaussian
    val b1 = (tb > hiddenDeim).randGaussian
    val w2 = (tb > hiddenDeim ^ labelDim).randGaussian
    val b2 = (tb > labelDim).randGaussian

//    gradcheck_naive[InputDim~HiddenNum, (Int,Int)]((x: Tensor[InputDim~HiddenNum]) => forward_backward_prop(data, labels, x, b1, w2, b2), w1)
  }

  def main(args: Array[String]) {
    sanity_check()
  }






}
