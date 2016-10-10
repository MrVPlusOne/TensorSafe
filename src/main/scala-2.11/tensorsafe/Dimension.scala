package tensorsafe

import scala.annotation.implicitNotFound

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
  def dim: Int
}

object DimValue{
  def const[D](v: Int) = new DimValue[D] {
    override def dim: Int = v
  }

  val unitDim = const[UnitDim](1)
}


class ShapeValue[S]private (shape: IndexedSeq[Int]){
  def append[D<:Dimension](d: DimValue[D]) = new ShapeValue[(S,D)](shape:+d.dim)
}

object ShapeValue{
  def single[D<:Dimension](d: DimValue[D]) = new ShapeValue[D](IndexedSeq(d.dim))
}

/**
 * A statement that either one of 'A' and 'B' is UnitDim, or 'A' == 'B'
 * @tparam A
 * @tparam B
 * @tparam R The bigger one of 'A' and 'B'
 */
trait DimMatch[A,B,R]

/**
 * A statement that tensor shape 'A' and tensor shape 'B' can be used in broadcast operations like '+' and '*'
 * @tparam A tensor shape 'A'
 * @tparam B tensor shape 'B'
 * @tparam R shape of the result tensor
 */
@implicitNotFound(msg = "Cannot find a type class for broadcast operation between ${A} and ${B}")
trait ShapeBroadcast[A,B,R]

/**
 * A tensor of shape 'A' matrix multiply a tensor of shape 'B' should results in a tensor of shape 'C'
 */
trait DimMul[A,B,R]

/**
 * A tensor of shape 'B' is the transpose of the tensor of shape 'A'
 */
trait ShapeReverse[A,R]

/**
 * A TNumber can be used to index a dimension of Shape S, resulting R if delete this dimension
 */
trait InRange[N<:TNumber, S, R]

case class TNumberValue[N<:TNumber](value: Int)

