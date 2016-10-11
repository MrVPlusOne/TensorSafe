package tensorsafe.test.example

import tensorsafe._
import tensorsafe.Imports._

trait DataNum extends VarDim
trait InputDim extends VarDim
trait LabelNum extends VarDim
trait HiddenNum extends VarDim

object NeuralNetworkExample {

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
  def gradcheck_naive[S,A,Idx](f: Tensor[S]=> (Double, Tensor[S]), x: Tensor[S])(implicit s2i: ShapeToIndex[S,Idx], i2v: IndexToVec[Idx]): Unit = {
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
      import math.{max,abs}
      val relativeDiff =  abs(numericGrad-grad(i)) / max(max(1, abs(numericGrad)), abs(grad(i)))

      if(relativeDiff > 1e-5){
        println("Gradient check failed.")
        println(s"First gradient error found at index $i")
        println(s"Your gradient: ${grad(i)} \t Numerical gradient: $numericGrad")
      }
    })

    println("Gradient check passed!")
  }










}
