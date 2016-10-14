package tensorsafe

import TensorOps._

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


/**
 * Created by weijiayi on 09/10/2016.
 */
class Tensor[S] private (private val ndArray: INDArray) {
  type Shape = S
  type Type = Tensor[Shape]

  def shape = ndArray.shape()

  def shapeIndexedSeq = shape.toIndexedSeq

  private def binaryOp[S1](t1: Tensor[S1], op: (INDArray,INDArray) => INDArray)
                          (implicit sb: ShapeBroadcast[S,S1]): Tensor[sb.Out] = {
    val newShape = broadCastShape(shape, t1.shape)
    def broadCastAndShape(a: INDArray): INDArray = {
      val reshaped = if(a.shape().length<newShape.length) {
        val rs = IndexedSeq.fill(newShape.length - a.shape().length)(1) ++ a.shape()
        a.reshape(rs: _*)
      } else a
      reshaped.broadcast(newShape :_*)
    }
    val array = op(broadCastAndShape(ndArray),broadCastAndShape(t1.ndArray))
    new Tensor[sb.Out](array)
  }

  //--- binary broadcast operations
  def *^[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Tensor[sb.Out] = binaryOp(t1, _ mul _)

  def *^=[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Unit = binaryOp(t1, _ muli _)(sb)

  def /[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Tensor[sb.Out] = binaryOp[S1](t1, _ div _)

  def /=[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Unit = binaryOp(t1, _ divi _)(sb)

  def +[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Tensor[sb.Out] = binaryOp[S1](t1, _ add _)

  def +=[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Unit = binaryOp(t1, _ addi _)(sb)

  def -[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Tensor[sb.Out] = binaryOp[S1](t1, _ sub _)

  def -=[S1](t1: Tensor[S1])(implicit sb: ShapeBroadcast[S,S1]): Unit = binaryOp(t1, _ subi _)(sb)


  //--- matrix multiplication
  def *[S1](t1: Tensor[S1])(implicit mmul: MatMul[S,S1]): Tensor[mmul.Out] = new Tensor[mmul.Out](ndArray mmul t1.ndArray)

  def *=[S1](t1: Tensor[S1])(implicit mmul: MatMul[S,S1]): Unit = ndArray mmuli t1.ndArray


  //--- transformations
  /** inefficient */
  def mapAll(f: Double => Double): Tensor[S] = {
    val d = duplicate
    d.vectorIndices.foreach(i => d.ndArray.putScalar(i.toArray, f(d.ndArray.getDouble(i:_*))))
    d
  }

  /** inefficient */
  def foreachAll(f: Double => Double): Unit = vectorIndices.foreach(i => f(ndArray.getDouble(i:_*)))

  def sum[Axis<:TNumber, NewShape](axis: Axis)(implicit inRange: InRange[Axis, S, NewShape], tv: TNumberValue[Axis]): Tensor[NewShape] = {
    val dim = rank-1-tv.value
    new Tensor[NewShape](ndArray.sum(dim))
  }


  //--- direct operations
  def t(implicit rev: RListOps.Reverse[S]): Tensor[rev.Out] = new Tensor[rev.Out](ndArray.transpose())

  def duplicate = new Tensor[S](ndArray.dup())

  def rank = shape.length

  def sumAll: Double = ndArray.sumNumber().doubleValue()

  def indices[Idx](implicit s2i: ShapeToIndex[S,Idx]): Stream[Idx] = {
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

  def size = shape.product

  def data = ndArray.data().asDouble()

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
    * @param flattened all the elements in this tensor, row-major
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
  def > [D<:Dimension](dv: DimValue[D]) = {
    implicit val dImplicit = dv
    new TensorBuilder[RNil~D](implicitly[ShapeValue[RNil~D]])
  }

  def scalar(x: Double) = (TensorBuilder > unitDim).create(Array(x))

  def apply[S](implicit sv: ShapeValue[S]) = new TensorBuilder[S](sv)
}
