import tensorsafe.DimValue._
import tensorsafe.TensorBuilder._
import tensorsafe._
import tensorsafe.Implicits._

trait Dim2 extends VarDim

trait Dim3 extends VarDim

trait Dim4 extends VarDim

trait Dim5 extends VarDim

trait Dim6 extends VarDim


val dim2 = const[Dim2](2)
val dim3 = const[Dim3](3)
val dim4 = const[Dim4](4)
val dim5 = const[Dim5](5)
val dim6 = const[Dim6](6)

val t1 = (vector(dim2)^ dim3 ^ dim4 ^ unitDim).zeros
val t2 = (vector(dim3) ^ unitDim ^ dim5).ones

t1 + t2

val t3 = (vector(dim2) ^ dim3).rand
t3
t3.t

