package tensorsafe.example

import tensorsafe._

trait DataNumber extends VarDim

trait DataDimension extends VarDim

trait FeatureDimension extends VarDim

object BasicExample {

  def main(args: Array[String]): Unit = {
    import DimValue.const

    implicit val dataNumber = const[DataNumber](10)
    implicit val dataDimension = const[DataDimension](3)
    implicit val featureDimension = const[FeatureDimension](6)

    val inputData = TensorBuilder[RNil~DataNumber~DataDimension].ones
    val weights = TensorBuilder[RNil~DataDimension~FeatureDimension].rand

    val bias = TensorBuilder[RNil~FeatureDimension].randGaussian
    val featureVectors = inputData * weights + bias

    val t1 = TensorBuilder[RNil~DataNumber~UnitDim~FeatureDimension].zeros // UnitDim is a dimension with value 1
    val t2 = TensorBuilder[RNil~DataDimension~UnitDim].ones
    val t3 = t1 + t2 // t3 has type Tensor[RNil~DataNumber~DataDimension~FeatureDimension]

    println(featureVectors)

  }

}
