package tensorsafe

import tensorsafe.ParameterList.TensorList
import tensorsafe.TensorOps.ShapeValue

/**
 * implicits needed for type level computations
 */

object Implicits {
  // TNumber to int
  implicit val T0Value = TNumberValue[T0](0)

  implicit def TNValue[N <: TNumber](implicit v: TNumberValue[N]) = TNumberValue[TN[N]](v.value+1)

  // TNumber for indexing Shape
  private[this] val singleton_inRange = new InRange[Any,Any,Any] {}


  implicit def inRangeBase2[D <: Dimension, S]: InRange[T0, (S, D), S] = cast(singleton_inRange)

  implicit def inRangeTn1[D1 <: Dimension, D2 <: Dimension]: InRange[TNumber.t1, (D1, D2), D2] = cast(singleton_inRange)

  implicit def inRangeRec2[D <: Dimension, N <: TNumber, S, S1, C](implicit i: InRange[N, S, S1]): InRange[TN[N], (S, D), (S1, D)] = cast(singleton_inRange)

}
