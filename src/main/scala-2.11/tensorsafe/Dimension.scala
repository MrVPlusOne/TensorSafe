package tensorsafe

import scala.annotation.implicitNotFound

sealed trait RNil {
  def ~[A](a: A) = (this, a)
}

case object RNil extends RNil

/**
 * A type to represent a dimension, should only take positive integer value
 */
sealed trait Dimension

/**
 * A dimension of 1
 */
sealed trait UnitDim extends Dimension

trait VarDim extends Dimension

/**
 * Provide the integer value of a Dimension
 * @tparam D Dimension type
 */
trait DimValue[D]{
  def value: Int
}

object DimValue{
  def const[D](v: Int) = new DimValue[D] {
    override def value: Int = v
  }

}


/**
  * A statement that either one of 'A' and 'B' is UnitDim, or 'A' == 'B', 'R' is bigger one of 'A' and 'B'
  */
trait DimMatch[A,B]{
  type R
}

object DimMatch{
  type Aux[A,B,R0] = DimMatch[A,B]{type R = R0}

  val singleton = new DimMatch[Any,Any] { type R = Any }

  implicit def dimMatch_unit: Aux[UnitDim, UnitDim, UnitDim] = cast(singleton)

  implicit def dimMatch_rightUnit[D <: Dimension]: Aux[D, UnitDim, D] = cast(singleton)

  implicit def dimMatch_leftUnit[D <: Dimension]: Aux[UnitDim, D, D] = cast(singleton)

  implicit def dimMatch_self[D]: Aux[D, D, D] = cast(singleton)

  object tests{
    implicitly[Aux[UnitDim,UnitDim,UnitDim]]
  }
}



/**
 * A TNumber can be used to index a dimension of Shape S, resulting R if delete this dimension
 */
trait InRange[N, S, R]

case class TNumberValue[N<:TNumber](value: Int)
