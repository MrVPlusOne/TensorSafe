import org.nd4j.linalg.factory.Nd4j
import tensorsafe.DimValue._
import tensorsafe._
import tensorsafe.Implicits._
import tensorsafe.TensorOps.ShapeValue

trait Dim2 extends VarDim
trait Dim3 extends VarDim
trait Dim4 extends VarDim
trait Dim5 extends VarDim
trait Dim6 extends VarDim


implicit val dim2 = const[Dim2](2)
implicit val dim3 = const[Dim3](3)
implicit val dim4 = const[Dim4](4)
implicit val dim5 = const[Dim5](5)
implicit val dim6 = const[Dim6](6)

val t1 = tb.apply[RNil~Dim2~Dim3~Dim4~UnitDim].randGaussian
val t2 = (tb > (dim3) ^ unitDim ^ dim5).ones

123.456*2

val ppp = t1 + t2 + t1
val pt = ppp.t
val t1t = t1.t

scalar(1.0)


val t3 = (tb > (dim2) ^ dim3).rand
t3+t3
t3.t


implicitly[DimValue[UnitDim]]
implicitly[ShapeValue[RNil~Dim2~Dim3~Dim4~UnitDim]].shape
val list = ParameterList.tensorsToParams(RNil~t1~t2)

//ParameterList.paramsToTensors[RNil~Tensor[t1Shape]~t2Type](list)

val ((_, tr1), tr2) = ParameterList.paramsToTensors[RNil~t1.Type~t2.Type](list)
tr1+tr2
//val tr3 = tr1 *^ tr2
