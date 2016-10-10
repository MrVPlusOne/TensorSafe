package tensorsafe.test.example

import tensorsafe.{TNumber, VarDim, Tensor}
import tensorsafe.Implicits._

trait DataNum extends VarDim
trait InputDim extends VarDim
trait LabelNum extends VarDim
trait HiddenNum extends VarDim

object NeuralNetworkExample {

  def sigmoid(x: Double) = 1.0/(1+math.exp(-x))

  /**
   * Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
   */
  def forward_backward_prop(data: Tensor[(DataNum,InputDim)], labels: Tensor[(DataNum, LabelNum)],
      w1: Tensor[(InputDim, HiddenNum)], b1: Tensor[HiddenNum],
      w2: Tensor[(HiddenNum, LabelNum)], b2: Tensor[LabelNum]) = {


    // forward propagation
    val hidden = (data matMul w1 + b1).mapAll(sigmoid)
    val prediction = (hidden matMul w2) + b2
    val cost = -(prediction.mapAll(math.log) * labels).sumAll

    // backward propagation
    val delta = prediction - labels
    val gradW2 = hidden.t matMul delta
    val gradb2 = delta.sum(TNumber.n0)
  }
}
