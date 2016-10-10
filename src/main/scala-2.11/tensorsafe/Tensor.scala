package tensorsafe


/**
 * Created by weijiayi on 09/10/2016.
 */
trait Tensor[Shape] {
  def * [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[R]

  def + [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[Shape]

  def - [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[Shape]

  def matMul[D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Tensor[R]

  def shape: IndexedSeq[Int]

  def mapAll(f: Double => Double): Tensor[Shape]

  def sumAll: Double

  def t[TShape](implicit trans: ShapeReverse[Shape, TShape]): Tensor[TShape]

  def sum[Axis<:TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, Shape, NewShape]): Tensor[NewShape]
}

trait TensorImpl {
  def build[Shape](p: ShapeValue[Shape]): Tensor[Shape]
}

class TensorBuilder[Shape] private (val p: ShapeValue[Shape]){
  def nextDim[D<:Dimension](d: DimValue[D]) = new TensorBuilder(p.append(d))

  def ^ [D<:Dimension](d: DimValue[D]) = nextDim(d)

  def build(implicit impl: TensorImpl): Tensor[Shape] = impl.build(p)

  def ^! (implicit impl: TensorImpl) = build(impl)
}

object TensorBuilder{
  def vector[D<:Dimension](d: DimValue[D]) = new TensorBuilder(ShapeValue.single(d))
}
