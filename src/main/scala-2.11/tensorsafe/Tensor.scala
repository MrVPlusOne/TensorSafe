package tensorsafe

import TensorOps._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


/**
 * Tensor with its shape injected into type parameter 'S'. Use ND4j's ndArray as its internal representation.
 */
class Tensor[S] private (private val ndArray: INDArray) {
  /** an easy access to the shape 'S' */
  type Shape = S
  /** an easy access to the type 'Tensor[S]' */
  type Type = Tensor[Shape]

  def shape = ndArray.shape()

  /**
    * A binary operator with broadcasting behavior
    * @param t1 the other tensor
    * @param op the underlying ndArray calculation
    * @param sb two tensors have compatible shapes
    */
  private def binaryOp[S1, Out](t1: Tensor[S1], op: (INDArray,INDArray) => INDArray)
                          (implicit sb: ShapeBroadcast[S,S1, Out]): Tensor[Out] = {
    val newShape = broadCastShape(shape, t1.shape)
    def broadCastAndShape(a: INDArray): INDArray = {
      val reshaped = if(a.shape().length<newShape.length) {
        val rs = IndexedSeq.fill(newShape.length - a.shape().length)(1) ++ a.shape()
        a.reshape(rs: _*)
      } else a
      reshaped.broadcast(newShape :_*)
    }
    val array = op(broadCastAndShape(ndArray),broadCastAndShape(t1.ndArray))
    new Tensor[Out](array)
  }

  //--- binary broadcast operations
  /** element-wise multiplication */
  def *^[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Tensor[Out] = binaryOp(t1, _ mul _)

  /** in place element-wise multiplication */
  def *^=[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Unit = binaryOp(t1, _ muli _)(sb)

  /** element-wise division */
  def /[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Tensor[Out] = binaryOp(t1, _ div _)

  /** in place element-wise division */
  def /=[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Unit = binaryOp(t1, _ divi _)(sb)

  /** element-wise plus */
  def +[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Tensor[Out] = binaryOp(t1, _ add _)

  /** in place element-wise plus */
  def +=[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Unit = binaryOp(t1, _ addi _)(sb)

  /** element-wise subtraction */
  def -[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Tensor[Out] = binaryOp(t1, _ sub _)

  /** in place element-wise plus */
  def -=[S1,Out](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1,Out]): Unit = binaryOp(t1, _ subi _)(sb)


  //--- equality
  def allZero(threshold: Double = 1e-6): Boolean = math.abs(sumAll)/size <= threshold

  def =~=[S1,Out](t1: Tensor[S1], threshold: Double = 1e-6)(implicit sb: ShapeBroadcast[S,S1,Out]): Boolean = (this - t1).allZero(threshold)

  //--- matrix multiplication
  /** matrix multiplication */
  def *[S1,Out](t1: Tensor[S1])(implicit mmul: MatMul[S,S1,Out]): Tensor[Out] = new Tensor[Out](ndArray mmul t1.ndArray)

  /** in place matrix multiplication */
  def *=[S1,Out](t1: Tensor[S1])(implicit mmul: MatMul[S,S1,Out]): Unit = ndArray mmuli t1.ndArray


  //--- transformations
  /** inefficient */
  def mapAll(f: Double => Double): Tensor[S] = {
    val d = duplicate
    d.vectorIndices.foreach(i => d.ndArray.putScalar(i.toArray, f(d.ndArray.getDouble(i:_*))))
    d
  }

  /** inefficient */
  def foreachAll(f: Double => Double): Unit = vectorIndices.foreach(i => f(ndArray.getDouble(i:_*)))

  /**
    * use summation to reduce the rank of this tensor
    * @param axis which axis to sum over, should be a TNumber, smaller axis corresponds to inner dimension,
    *             check the resulting shape
    */
  def sum[Axis<:TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, S, NewShape], tv: TNumberValue[Axis]): Tensor[NewShape] = {
    val dim = rank-1-tv.value
    new Tensor[NewShape](ndArray.sum(dim))
  }


  //--- direct operations
  /** tensor transpose
    * e.g. a tensor of shape [RNil~A~B~C] becomes [RNil~C~B~A]*/
  def t(implicit rev: RListOps.Reverse[S]): Tensor[rev.Out] = new Tensor[rev.Out](ndArray.transpose())

  def duplicate = new Tensor[S](ndArray.dup())

  def rank = shape.length

  /** the sum of all elements */
  def sumAll: Double = ndArray.sumNumber().doubleValue()

  /** return a lazy sequence of type-safe indices of all elements */
  def indices[Idx](implicit s2i: ShapeToIndex[S,Idx]): Stream[Idx] = {
    val s = shape
    def indicesFromShape(s: IndexedSeq[Int]): Stream[Any] = {
      if(s.isEmpty) Stream.empty[Int]
      else (0 until s.last).toStream.flatMap(h => indicesFromShape(s.init).map(i => (h, i)))
    }
    indicesFromShape(s).asInstanceOf[Stream[Idx]]
  }

  /** return a lazy sequence of Vector[Int] indices of all elements */
  private def vectorIndices: Stream[Vector[Int]] = {
    val s = shape
    def indicesFromShape(s: IndexedSeq[Int]): Stream[Vector[Int]] = {
      if(s.isEmpty) Stream.empty
      else (0 until s.last).toStream.flatMap(h => indicesFromShape(s.init).map(i => i :+ h))
    }
    indicesFromShape(s)
  }


  //--- Access
  def apply[Idx](idx: Idx)(implicit s2i: ShapeToIndex[S,Idx], idx2Vec: IndexToVec[Idx]): Double = ndArray.getDouble(idx2Vec.vector(idx):_*)

  def update[Idx](idx: Idx, v: Double)(implicit s2i: ShapeToIndex[S,Idx], idx2Vec: IndexToVec[Idx]): Unit = ndArray.putScalar(idx2Vec.vector(idx).toArray, v)

  override def toString: String = {
    if(rank == 2){
      ndArray.toString
    }else{
      ndArray.toString
    }
  }

  /** the total number of elements this tensor contains */
  def size = shape.product

  /** all the elements of this tensor as an array */
  def data: Array[Double] = {
    val d = ndArray.data().asDouble()
    assert(d.length == size)
    d
  }

}

object Tensor {
  private def createUse[Shape](p: ShapeValue[Shape], single: Int => INDArray, plural: IndexedSeq[Int] => INDArray): Tensor[Shape] = {
    val shape = p.shape
    val array = if (shape.length == 1)
      single(shape.head)
    else plural(shape)
    new Tensor[Shape](array)
  }

  def zeros[Shape](p: ShapeValue[Shape]): Tensor[Shape] = createUse(p, Nd4j.zeros, Nd4j.zeros(_: _*))

  def ones[Shape](p: ShapeValue[Shape]): Tensor[Shape] = createUse(p, Nd4j.ones, Nd4j.ones(_: _*))

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
    *
    * @param flattened all elements in this tensor, row-major
    */
  def create[Shape](flattened: Array[Double])(implicit p: ShapeValue[Shape]): Tensor[Shape] = {
    val array = if(p.shape.length == 1) Nd4j.create(flattened)
    else Nd4j.create(flattened, p.shape.toArray)
    new Tensor[Shape](array)
  }
}


class TensorBuilder[Shape] private (val sv: ShapeValue[Shape]){
  def nextDim[D<:Dimension](d: DimValue[D]) = new TensorBuilder(sv.append(d))

  private def impl = Tensor

  def ^ [D<:Dimension](d: DimValue[D]) = nextDim(d)

  def zeros: Tensor[Shape] = impl.zeros(sv)

  def ones: Tensor[Shape] = impl.ones(sv)

  /**
   * create a tensor with shape 'Shape' from a java array.
   * @param flattened all the elements in this tensor, row-major
   */
  def create(flattened: Array[Double]): Tensor[Shape] = impl.create(flattened)(sv)

  /**
   * generate uniform random numbers in the range 0 to 1
   */
  def rand: Tensor[Shape] = impl.rand(sv)

  /**
   * generate Gaussian random numbers with mean zero and standard deviation 1
   */
  def randGaussian: Tensor[Shape] = impl.randGaussian(sv)
}


object TensorBuilder{
  /** Create a tensor builder.
    * Example usage:
    *   (TensorBuilder > dim1 ^^ dim2 ^^ dim3).zeros
    *  */
  def > [D<:Dimension](dv: DimValue[D]) = {
    implicit val dImplicit = dv
    new TensorBuilder[RNil~D](implicitly[ShapeValue[RNil~D]])
  }

  /** Wrap a Double value into a vector(1D tensor) */
  def scalar(x: Double) = (TensorBuilder > unitDim).create(Array(x))

  /** construct a `TensorBuilder` with shape `S` */
  def apply[S](implicit sv: ShapeValue[S]) = new TensorBuilder[S](sv)
}
