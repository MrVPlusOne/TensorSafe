package tensorsafe.example

import tensorsafe._

trait DataNumber extends VarDim

trait DataDimension extends VarDim

trait FeatureDimension extends VarDim

object BasicExample {

  def main(args: Array[String]): Unit = {
    val inputData = TensorBuilder[RNil~DataNumber~DataDimension].randGaussian
  }

}
