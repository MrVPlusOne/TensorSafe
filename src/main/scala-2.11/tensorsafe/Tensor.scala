package tensorsafe

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.BroadcastOp
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp
import org.nd4j.linalg.factory.Nd4j


/**
 * Created by weijiayi on 09/10/2016.
 */
class Tensor[Shape] private (private val ndArray: INDArray) {

  def shape = ndArray.shape()

  def *^ [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[R] = {
    new Tensor[R](ndArray mul t1.ndArray)
  }

  def *^= [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Unit = new Tensor[R](ndArray muli t1.ndArray)

  def + [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[R] = new Tensor[R](ndArray add t1.ndArray)

  def += [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Unit = new Tensor[R](ndArray addi t1.ndArray)

  def - [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[R] = new Tensor[R](ndArray sub t1.ndArray)

  def -= [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Unit = new Tensor[R](ndArray subi t1.ndArray)

  def matMul[D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Tensor[R] = new Tensor[R](ndArray mmul t1.ndArray)

  def matMulInplace[D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Unit = new Tensor[R](ndArray mmuli t1.ndArray)

  def * [D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Tensor[R] = matMul(t1)

  def *=[D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Unit = matMulInplace(t1)

  def rank: Int = shape.length

  def duplicate: Tensor[Shape] = new Tensor[Shape](ndArray.dup())

  /** inefficient */
  def mapAll(f: Double => Double): Tensor[Shape] = {
    val d = duplicate
    d.vectorIndices.foreach(i => d.ndArray.putScalar(i.toArray, f(d.ndArray.getDouble(i:_*))))
    d
  }

  /** inefficient */
  def foreachAll(f: Double => Double): Unit = vectorIndices.foreach(i => f(ndArray.getDouble(i:_*)))

  def indices[Idx](implicit s2i: ShapeToIndex[Shape,Idx]): Stream[Idx] = {
    val s = shape
    def indicesFromShape(s: IndexedSeq[Int]): Stream[Any] = {
      if(s.isEmpty) Stream.empty[Int]
      else (0 until s.last).toStream.flatMap(h => indicesFromShape(s.init).map(i => (h, i)))
    }
    indicesFromShape(s).asInstanceOf[Stream[Idx]]
  }

  private def vectorIndices: Stream[Vector[Int]] = {
    val s = shape
    def indicesFromShape(s: IndexedSeq[Int]): Stream[Vector[Int]] = {
      if(s.isEmpty) Stream.empty
      else (0 until s.last).toStream.flatMap(h => indicesFromShape(s.init).map(i => i :+ h))
    }
    indicesFromShape(s)
  }

  def sumAll: Double = ndArray.sumNumber().doubleValue()

  def t[TShape](implicit trans: ShapeReverse[Shape, TShape]): Tensor[TShape] = new Tensor[TShape](ndArray.transpose())

  def sum[Axis<:TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, Shape, NewShape], tv: TNumberValue[Axis]): Tensor[NewShape] = {
    val dim = rank-1-tv.value
    new Tensor[NewShape](ndArray.sum(dim))
  }

  def apply[Idx](idx: Idx)(implicit s2i: ShapeToIndex[Shape,Idx], idx2Vec: IndexToVec[Idx]): Double = ndArray.getDouble(idx2Vec.vector(idx):_*)

  def update[Idx](idx: Idx, v: Double)(implicit s2i: ShapeToIndex[Shape,Idx], idx2Vec: IndexToVec[Idx]): Unit = ndArray.putScalar(idx2Vec.vector(idx).toArray, v)

  override def toString: String = {
    if(rank == 2){
      ndArray.toString
    }else{
      ndArray.toString
    }
  }
}

object Tensor {


    def zeros[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new Tensor(Nd4j.zeros(p.shape :_*))

    def ones[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new Tensor(Nd4j.ones(p.shape :_*))

    /**
     * generate uniform random numbers in the range 0 to 1
     */
    def rand[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new Tensor(Nd4j.rand(p.shape.toArray))

    /**
     * generate Gaussian random numbers with mean zero and standard deviation 1
     */
    def randGaussian[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new Tensor(Nd4j.randn(p.shape.toArray))

    /**
     * create a tensor with shape 'Shape' from a java array.
     * @param flattened all the elements in this tensor, row-major
     */
    def create[Shape](flattened: Array[Double])(p: ShapeValue[Shape]): Tensor[Shape] =
      new Tensor(Nd4j.create(flattened.toArray))

}


class TensorBuilder[Shape] private (val p: ShapeValue[Shape]){
  def nextDim[D<:Dimension](d: DimValue[D]) = new TensorBuilder(p.append(d))

  private def impl = Tensor

  def ^ [D<:Dimension](d: DimValue[D]) = nextDim(d)

  def zeros: Tensor[Shape] = impl.zeros(p)

  def ones: Tensor[Shape] = impl.ones(p)

  /**
   * create a tensor with shape 'Shape' from a java array.
   * @param flattened all the elements in this tensor, row-major
   */
  def create(flattened: Array[Double]): Tensor[Shape] = impl.create(flattened)(p)

  /**
   * generate uniform random numbers in the range 0 to 1
   */
  def rand: Tensor[Shape] = impl.rand(p)

  /**
   * generate Gaussian random numbers with mean zero and standard deviation 1
   */
  def randGaussian: Tensor[Shape] = impl.randGaussian(p)
}

object TensorBuilder{
  def vector[D<:Dimension](d: DimValue[D]) = new TensorBuilder(ShapeValue.single(d))
}
