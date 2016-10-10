package tensorsafe

/**
 * Created by weijiayi on 10/10/2016.
 */
class ND4JTensor[S] extends Tensor[S]{
  override def *^[S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[S, S1, R]): Tensor[R] = ???

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
    override def build[Shape](p: ShapeValue[Shape]): Tensor[Shape] = ???
  }
}