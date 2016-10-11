package tensorsafe

/**
 * implicits needed for type level computations
 */

object Implicits {
  implicit val unitDimProvider = DimValue.const[UnitDim](1)

  // DimMatch
  private[this] val singleton_dimMatch = new DimMatch[Any,Any,Any] {}

  implicit def dimMatch_rightUnit[D <: Dimension]: DimMatch[D, UnitDim, D] = cast(singleton_dimMatch)

  implicit def dimMatch_leftUnit[D <: VarDim]: DimMatch[UnitDim, D, D] = cast(singleton_dimMatch)

  implicit def dimMatch[D <: VarDim]: DimMatch[D, D, D] = cast(singleton_dimMatch)

  // Tensor Shape Broadcast
  private[this] val singleton_shapeBroadcast = new ShapeBroadcast[Any,Any,Any] {}
  
  implicit def shapeBroadcast1[D1, D2, R](implicit m1: DimMatch[D1, D2, R]): ShapeBroadcast[D1, D2, R] = cast(singleton_shapeBroadcast)

  implicit def shapeBroadcastShortLong[D1 <: Dimension, DD2, D2, R](implicit m: DimMatch[D1, D2, R]): ShapeBroadcast[D1, (DD2, D2), (DD2, R)] = cast(singleton_shapeBroadcast)

  implicit def shapeBroadcastLongShort[D1 <: Dimension, DD2, D2, R](implicit m: DimMatch[D1, D2, R]): ShapeBroadcast[(DD2, D2), D1, (DD2, R)] = cast(singleton_shapeBroadcast)

  implicit def shapeBroadcastND[D1Init, D1Last, D2Init, D2Last, RInit, RLast](implicit m1: DimMatch[D1Last, D2Last, RLast], m2: ShapeBroadcast[D1Init, D2Init, RInit]): ShapeBroadcast[(D1Init, D1Last), (D2Init, D2Last), (RInit, RLast)] = cast(singleton_shapeBroadcast)


  // Matrix multiplication
  private[this] val singleton_shapeMul = new ShapeMul[Any,Any,Any] {}

  implicit def shapeMul[D1, D2, D3]: ShapeMul[(D1, D2), (D2, D3), (D1, D3)] = cast(singleton_shapeMul)

  // Shape Reverse
  trait ReverseLoop[A, Left, Acc]
  private[this] val singleton_reverseLoop = new ReverseLoop[Any,Any,Any] {}
  private[this] val singleton_shapeReverse = new ShapeReverse[Any,Any] {}


  implicit def reverseBase[A, L <: Dimension, R](implicit l: ReverseLoop[A, L, R]): ShapeReverse[A, (R, L)] = cast(singleton_shapeReverse)

  implicit def reverseLoop[Init, L, A, Acc](implicit l: ReverseLoop[A, (Init, L), Acc]): ReverseLoop[A, Init, (Acc, L)] = cast(singleton_reverseLoop)

  implicit def reverseLoopStart[S, A]: ReverseLoop[(S, A), S, A] = cast(singleton_reverseLoop)

  // TNumber to int
  implicit val T0Value = TNumberValue[T0](0)

  implicit def TNValue[N <: TNumber](implicit v: TNumberValue[N]) = TNumberValue[TN[N]](v.value+1)

  // TNumber for indexing Shape
  private[this] val singleton_inRange = new InRange[Any,Any,Any] {}


  implicit def inRangeBase2[D <: Dimension, S]: InRange[T0, (S, D), S] = cast(singleton_inRange)

  implicit def inRangeTn1[D1 <: Dimension, D2 <: Dimension]: InRange[TNumber.t1, (D1, D2), D2] = cast(singleton_inRange)

  implicit def inRangeRec2[D <: Dimension, N <: TNumber, S, S1, C](implicit i: InRange[N, S, S1]): InRange[TN[N], (S, D), (S1, D)] = cast(singleton_inRange)

  // Shape Index as tuples
  private[this] val singleton_s2i = new ShapeToIndex[Any,Any] {}


  implicit def indexVector[D<:Dimension]: ShapeToIndex[D, Int] = cast(singleton_s2i)

  implicit def indexTensor[D<:Dimension,S,R1](implicit r1: ShapeToIndex[S,R1]): ShapeToIndex[(S,D),(R1,Int)] = cast(singleton_s2i)

  // Tuple Index to Vector
  implicit val index2VecBase: IndexToVec[Int] = new IndexToVec[Int] {
    override def vector(i: Int): Vector[Int] = Vector(i)
  }

  implicit def index2VecRec[Init](implicit i2v: IndexToVec[Init]): IndexToVec[(Init,Int)] = new IndexToVec[(Init, Int)] {
    override def vector(i: (Init, Int)): Vector[Int] = i2v.vector(i._1) :+ i._2
  }
}
