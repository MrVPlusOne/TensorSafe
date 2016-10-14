package tensorsafe.example

import tensorsafe.Implicits._
import tensorsafe._
import tensorsafe.TensorOps._

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
  def forward_backward_prop(data: Tensor[RNil~DataNum~InputDim], labels: Tensor[RNil~DataNum~LabelNum],
      w1: Tensor[RNil~InputDim~HiddenNum], b1: Tensor[RNil~HiddenNum],
      w2: Tensor[RNil~HiddenNum~LabelNum], b2: Tensor[RNil~LabelNum]) = {


    // forward propagation
    val hidden = (data * w1 + b1).mapAll(sigmoid)
    val prediction = (hidden * w2) + b2
    val cost = -(prediction.mapAll(math.log) *^ labels).sumAll

    // backward propagation
    val delta = prediction - labels
    val gradW2 = hidden.t * delta
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
  def gradcheck_naive(f: Array[Double] => (Double, Array[Double]), x: Array[Double]): Unit = {
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
        return
      }
    })

    println("Gradient check passed!")
  }



  trait D1 extends VarDim
  trait D2 extends VarDim
  trait D3 extends VarDim

  def sanity_check(): Unit = {
    def quad[S](implicit b: ShapeBroadcast.Aux[S,S,S], b2: ShapeBroadcast.Aux[S,RNil~UnitDim,S], sv: ShapeValue[S]) =
      (data: Array[Double]) => {
        val x = TensorBuilder[S].create(data)
        val y = (x *^ x).sumAll
        val dydx = (x *^ scalar(2)).data
        (y, dydx)
      }

    implicit val d1 = dim[D1](3)
    implicit val d2 = dim[D2](4)
    implicit val d3 = dim[D3](5)

    gradcheck_naive(quad[RNil~UnitDim], Array(123.456))
    gradcheck_naive(quad[RNil~D1], (TensorBuilder > d1).randGaussian.data)
    gradcheck_naive(quad[RNil~D2~D3], (TensorBuilder > d2 ^ d3).randGaussian.data)
  }

  def neural_sanity_check(random: Random): Unit = {
    implicit val dataDim = dim[DataNum](20)
    implicit val inputDim = dim[InputDim](10)
    implicit val hiddenDim = dim[HiddenNum](5)
    implicit val labelDim = dim[LabelNum](10)

    val data = (tb > dataDim ^ inputDim).randGaussian
    val labels = (tb > dataDim ^ labelDim).zeros

    for(i <- 0 until dataDim.value){
      labels(RNil~i~random.nextInt(labelDim.value)) = 1
    }

    val w1 = (tb > inputDim ^ hiddenDim).randGaussian
    val b1 = (tb > hiddenDim).randGaussian
    val w2 = (tb > hiddenDim ^ labelDim).randGaussian
    val b2 = (tb > labelDim).randGaussian

    def f(x: Array[Double]) = {
      val ((((_, w1$), b1$), w2$), b2$) = ParameterList.paramsToTensors[RNil~w1.Type~b1.Type~w2.Type~b2.Type](x)
      val (cost, w1g,b1g,w2g,b2g) = forward_backward_prop(data, labels, w1$, b1$, w2$, b2$)
      (cost, ParameterList.tensorsToParams(RNil~w1g~b1g~w2g~b2g).toArray)
    }

    gradcheck_naive(f, ParameterList.tensorsToParams(RNil~w1~b1~w2~b2).toArray)
  }

  def main(args: Array[String]) {
    sanity_check()

    neural_sanity_check(new Random())
  }






}
