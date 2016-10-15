package tensorsafe

import scala.annotation.implicitNotFound

/**
  * Created by weijiayi on 14/10/2016.
  */
object TensorOps {

  @implicitNotFound(msg = "Not enough value information for Shape '${S}'\n Please make sure that each " +
    "dimension has a corresponding DimValue in scope. Zero-rank Tensor is not allowed.")
  class ShapeValue[S]private (val shape: IndexedSeq[Int]){
    def append[D<:Dimension](d: DimValue[D]) = new ShapeValue[(S,D)](shape:+d.value)
  }

  object ShapeValue{
//    def scalar = new ShapeValue[RNil](IndexedSeq())

    implicit def vector[D<:Dimension](implicit dv: DimValue[D]): ShapeValue[RNil~D] = new ShapeValue[RNil~D](IndexedSeq(dv.value))

    implicit def recursion[S,L](implicit sv: ShapeValue[S], dv: DimValue[L]): ShapeValue[S~L] =
      new ShapeValue[S~L](sv.shape :+ dv.value)
  }

  /**
    * A statement that tensor shape 'A' and tensor shape 'B' can be used in broadcast operations like '+' and '*'
    * @tparam A tensor shape 'A'
    * @tparam B tensor shape 'B'
    */
  @implicitNotFound(msg = "Cannot find a type class for broadcast operation between ${A} and ${B}")
  trait ShapeBroadcast[A,B, Out]

  object ShapeBroadcast{

    private[this] val singleton = new ShapeBroadcast[Any,Any, Any] { type Out = Any}

    implicit val nil: ShapeBroadcast[RNil, RNil, RNil] = cast(singleton)

    implicit def shortLong[S]: ShapeBroadcast[RNil, S, S] = cast(singleton)

    implicit def longShort[S]: ShapeBroadcast[S, RNil, S] = cast(singleton)

    implicit def recursion[L1,L2,RLast,Init1,Init2,RInit]
    (implicit m1: DimMatch.Aux[L1, L2, RLast], m2: ShapeBroadcast[Init1, Init2, RInit]):
    ShapeBroadcast[Init1~L1, Init2~L2, RInit~RLast] = cast(singleton)

  }

  /**
    * Matrix Multiplication
    */
  @implicitNotFound(msg = "Cannot find a implicit for matrix multiplication between ${A} and ${B}")
  trait MatMul[A,B, Out]

  object MatMul {
    private[this] val singleton = new MatMul[Any,Any,Any] {}

    implicit def mult2D[D1, D2, D3]: MatMul[RNil~D1~D2, RNil~D2~D3, RNil~D1~D3] = cast(singleton)
  }

  /**
    * A shape 'S' corresponds to index type 'I'
    */
  @implicitNotFound(msg="Index type ${I} does not match Shape type ${S}")
  trait ShapeToIndex[S,I]

  object ShapeToIndex{
    // Shape Index as tuples
    private[this] val singleton_s2i = new ShapeToIndex[Any,Any] {}


    implicit def indexScalar: ShapeToIndex[RNil, RNil] = cast(singleton_s2i)

    implicit def indexTensor[D,S,R1](implicit r1: ShapeToIndex[S,R1]): ShapeToIndex[(S,D),(R1,Int)] = cast(singleton_s2i)


  }

  /**
    * Convert A tuple index into a vector
    */
  trait IndexToVec[I]{
    def vector(i: I): Vector[Int]
  }

  object IndexToVec{
    // Tuple Index to Vector
    implicit val index2VecBase: IndexToVec[RNil] = new IndexToVec[RNil] {
      override def vector(x: RNil): Vector[Int] = Vector()
    }

    implicit def index2VecRec[Init](implicit i2v: IndexToVec[Init]): IndexToVec[(Init,Int)] = new IndexToVec[(Init, Int)] {
      override def vector(i: (Init, Int)): Vector[Int] = i2v.vector(i._1) :+ i._2
    }
  }
}
