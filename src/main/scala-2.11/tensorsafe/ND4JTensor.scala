package tensorsafe

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
 * Created by weijiayi on 10/10/2016.
 */
class ND4JTensor[S] private (val ndArray: INDArray) extends Tensor[S]{
  override def *^[S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[S, S1, R]): Tensor[R] =

  override def matMul[D1, R](t1: Tensor[D1])(implicit comp: DimMul[S, D1, R]): Tensor[R] = ???

  override def shape: IndexedSeq[Int] = ???

  override def +[S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[S, S1, R]): Tensor[S] = ???

  override def -[S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[S, S1, R]): Tensor[S] = ???

  override def mapAll(f: (Double) => Double): Tensor[S] = ???

  override def sumAll: Double = ???

  override def t[TShape](implicit trans: ShapeReverse[S, TShape]): Tensor[TShape] = ???

  override def sum[Axis <: TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, S, NewShape]): Tensor[NewShape] = ???

  override def apply[Idx](idx: Idx)(implicit i: ShapeToIndex[S, Idx]): Double = ???

  override def update[Idx](idx: Idx, v: Double)(implicit i: ShapeToIndex[S, Idx]): Unit = ???

  override def foreachAll(f: (Double) => Double): Unit = ???

  override def indices[Idx](implicit i: ShapeToIndex[S, Idx]): Seq[Idx] = ???
}

object ND4JTensor {
  implicit val impl = new TensorImpl {
    override def zeros[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new ND4JTensor(Nd4j.zeros(p.shape :_*))

    override def ones[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new ND4JTensor(Nd4j.ones(p.shape :_*))

    /**
     * generate uniform random numbers in the range 0 to 1
     */
    override def rand[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new ND4JTensor(Nd4j.rand(p.shape.toArray))

    /**
     * generate Gaussian random numbers with mean zero and standard deviation 1
     */
    override def randGaussian[Shape](p: ShapeValue[Shape]): Tensor[Shape] = new ND4JTensor(Nd4j.randn(p.shape.toArray))

    /**
     * create a tensor with shape 'Shape' from a java array.
     * @param flattened all the elements in this tensor, row-major
     */
    override def create[Shape](flattened: Array[Double])(p: ShapeValue[Shape]): Tensor[Shape] =
      new ND4JTensor(Nd4j.create(flattened, p.shape.toArray))
  }
}