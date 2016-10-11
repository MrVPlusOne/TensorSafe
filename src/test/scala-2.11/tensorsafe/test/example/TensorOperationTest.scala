package tensorsafe.test.example

import tensorsafe.Implicits._
import tensorsafe.DimValue._
import tensorsafe.TensorBuilder._
import tensorsafe._

/**
 * Created by weijiayi on 11/10/2016.
 */
object TensorOperationTest {

  trait D2 extends VarDim

  trait D3 extends VarDim

  trait D4 extends VarDim

  trait D5 extends VarDim

  trait D6 extends VarDim



  def main(args: Array[String]) {
    val dim2 = const[D2](2)
    val dim3 = const[D3](3)
    val dim4 = const[D4](4)
    val dim5 = const[D5](5)
    val dim6 = const[D5](6)

    val t1 = (TensorBuilder > dim2 ^ dim3 ^ dim4 ^ unitDim).zeros
    val t2 = (TensorBuilder > dim3 ^ unitDim ^ dim5).ones

    t1 + t2
    t1 *^ t2
  }
}

