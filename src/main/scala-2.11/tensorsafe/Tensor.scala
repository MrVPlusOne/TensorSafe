package tensorsafe


/**
 * Created by weijiayi on 09/10/2016.
 */
trait Tensor[Shape] {
  def *^ [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[R]

  def + [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[Shape]

  def - [S1, R](t1: Tensor[S1])(implicit comp: ShapeBroadcast[Shape,S1,R]): Tensor[Shape]

  def matMul[D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Tensor[R]

  def * [D1,R](t1: Tensor[D1])(implicit comp: DimMul[Shape,D1,R]): Tensor[R] = matMul(t1)

  def shape: IndexedSeq[Int]

  def mapAll(f: Double => Double): Tensor[Shape]

  def foreachAll(f: Double => Double): Unit

  def indices[Idx](implicit s2i: ShapeToIndex[Shape,Idx]): Seq[Idx]

  def sumAll: Double

  def t[TShape](implicit trans: ShapeReverse[Shape, TShape]): Tensor[TShape]

  def sum[Axis<:TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, Shape, NewShape]): Tensor[NewShape]

  def apply[Idx](idx: Idx)(implicit s2i: ShapeToIndex[Shape,Idx]): Double

  def update[Idx](idx: Idx, v: Double)(implicit s2i: ShapeToIndex[Shape,Idx]): Unit
}

trait TensorImpl {
  def zeros[Shape](p: ShapeValue[Shape]): Tensor[Shape]
  def ones[Shape](p: ShapeValue[Shape]): Tensor[Shape]

  /**
   * create a tensor with shape 'Shape' from a java array.
   * @param flattened all the elements in this tensor, row-major
   */
  def create[Shape](flattened: Array[Double])(p: ShapeValue[Shape]): Tensor[Shape]

  /**
   * generate uniform random numbers in the range 0 to 1
   */
  def rand[Shape](p: ShapeValue[Shape]): Tensor[Shape]

  /**
   * generate Gaussian random numbers with mean zero and standard deviation 1
   */
  def randGaussian[Shape](p: ShapeValue[Shape]): Tensor[Shape]
}

class TensorBuilder[Shape] private (val p: ShapeValue[Shape]){
  def nextDim[D<:Dimension](d: DimValue[D]) = new TensorBuilder(p.append(d))

  def ^ [D<:Dimension](d: DimValue[D]) = nextDim(d)

  def zeros(implicit impl: TensorImpl): Tensor[Shape] = impl.zeros(p)

  def ones(implicit impl: TensorImpl): Tensor[Shape] = impl.ones(p)

  /**
   * create a tensor with shape 'Shape' from a java array.
   * @param flattened all the elements in this tensor, row-major
   */
  def create(flattened: Array[Double])(implicit impl: TensorImpl): Tensor[Shape] = impl.create(flattened)(p)

  /**
   * generate uniform random numbers in the range 0 to 1
   */
  def rand(implicit impl: TensorImpl): Tensor[Shape] = impl.rand(p)

  /**
   * generate Gaussian random numbers with mean zero and standard deviation 1
   */
  def randGaussian(implicit impl: TensorImpl): Tensor[Shape] = impl.randGaussian(p)
}

object TensorBuilder{
  def vector[D<:Dimension](d: DimValue[D]) = new TensorBuilder(ShapeValue.single(d))
}
