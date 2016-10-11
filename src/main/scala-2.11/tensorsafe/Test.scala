package tensorsafe

import TensorBuilder.vector
import scala.language.postfixOps
import Imports._
import DimValue.const

/**
 * Created by weijiayi on 09/10/2016.
 */
object Test {

  trait Dim1 extends VarDim

  trait Dim2 extends VarDim

  trait Dim3 extends VarDim

  trait Dim4 extends VarDim

  trait Dim5 extends VarDim


  def main(args: Array[String]) {
    val dim1 = const[Dim1](5)
    val dim2 = const[Dim2](7)
    val dim4 = const[Dim4](4)
    val dim5 = const[Dim5](5)
    val dim3 = const[Dim3](3)

//    val t1 = (vector(dim1)^dim4 ^ dim2 ^ unitDim).zeros
//    val t2 = (vector(dim4) ^ unitDim ^ dim5).ones
//    val t3 = t1 *^ t2

    val t4 = (vector(dim1) ^ dim2).rand
    val t5 = (vector(unitDim) ^ dim2).randGaussian
    val t6 = t4 *^ t5

    println{
      t6.shape
    }

  }
}
