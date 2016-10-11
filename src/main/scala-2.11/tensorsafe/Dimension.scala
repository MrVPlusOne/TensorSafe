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
  def value: Int
}

object DimValue{
  def const[D](v: Int) = new DimValue[D] {
    override def value: Int = v
  }
}


class ShapeValue[S]private (val shape: IndexedSeq[Int]){
  def append[D<:Dimension](d: DimValue[D]) = new ShapeValue[(S,D)](shape:+d.value)
}

object ShapeValue{
  def single[D<:Dimension](d: DimValue[D]) = new ShapeValue[D](IndexedSeq(d.value))
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
@implicitNotFound(msg = "Cannot find a type class for matrix multiplication between ${A} and ${B}")
trait ShapeMul[A,B,R]

/**
 * A tensor of shape 'B' is the transpose of the tensor of shape 'A'
 */
trait ShapeReverse[A,R]

/**
 * A TNumber can be used to index a dimension of Shape S, resulting R if delete this dimension
 */
trait InRange[N, S, R]

case class TNumberValue[N<:TNumber](value: Int)

/**
 * A shape 'S' corresponds to index type 'I'
 */
@implicitNotFound(msg="Index type ${I} does not match Shape type ${S}")
trait ShapeToIndex[S,I]

/**
 * Convert A tuple index into a vector
 */
trait IndexToVec[I]{
  def vector(i: I): Vector[Int]
}