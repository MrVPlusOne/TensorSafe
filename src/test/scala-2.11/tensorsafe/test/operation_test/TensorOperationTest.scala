package tensorsafe.test.operation_test

import tensorsafe.DimValue._
import tensorsafe.{TensorBuilder, VarDim, _}

trait D2 extends VarDim

trait D3 extends VarDim

trait D4 extends VarDim

trait D5 extends VarDim

trait D6 extends VarDim

object TensorOperationTest {


  def main(args: Array[String]) {
    implicit val dim2 = const[D2](2)
    implicit val dim3 = const[D3](3)
    implicit val dim4 = const[D4](4)
    implicit val dim5 = const[D5](5)
    implicit val dim6 = const[D5](6)

    val t1 = (TensorBuilder > dim2 ^ unitDim).randGaussian
    val t2 = (TensorBuilder > unitDim ^ dim4).randGaussian
    val t3 = (tb > dim3 ^ dim2 ^ unitDim).ones
    val long = (tb > dim2 ^ dim3 ^ dim4 ^ dim5 ^ dim6 ^ dim2 ^ dim3 ^ dim4 ^ dim5 ^ dim5).ones

    val ll = long.t
    t1 *^ t2

    val t4 = (t1 + t2).t
    assert(TensorBuilder[t4.Shape].create(t4.data) =~= t4, "tensor reconstruct test")

  }
}
