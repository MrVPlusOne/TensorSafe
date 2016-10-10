package tensorsafe

/**
 * implicits needed for type level computations
 */

object Implicits {
  type ~[A, B] = (A, B)
  def ~[A,B](a: A, b: B) = (a,b)

  implicit val unitDimProvider = DimValue.const[UnitDim](1)

  implicit def dimMatch_rightUnit[D <: Dimension]: DimMatch[D, UnitDim, D] = neverRun

  implicit def dimMatch_leftUnit[D <: Dimension]: DimMatch[UnitDim, D, D] = neverRun

  implicit def dimMatch[D <: Dimension]: DimMatch[D, D, D] = neverRun

  // Tensor Shape Broadcast
  implicit def shapeBroadcast1[D1, D2 <: Dimension, R](implicit m1: DimMatch[D1, D2, R]): ShapeBroadcast[D1, D2, R] = neverRun

  implicit def shapeBroadcastShortLong[D1 <: Dimension, DD2, D2, R](implicit m: DimMatch[D1, D2, R]): ShapeBroadcast[D1, (DD2, D2), (DD2, R)] = neverRun

  implicit def shapeBroadcastLongShort[D1 <: Dimension, DD2, D2, R](implicit m: DimMatch[D1, D2, R]): ShapeBroadcast[(DD2, D2), D1, (DD2, R)] = neverRun

  implicit def shapeBroadcastND[D1Init, D1Last, D2Init, D2Last, RInit, RLast](implicit m1: DimMatch[D1Last, D2Last, RLast], m2: ShapeBroadcast[D1Init, D2Init, RInit]): ShapeBroadcast[(D1Init, D1Last), (D2Init, D2Last), (RInit, RLast)] = neverRun


  // Matrix multiplication
  implicit def shapeMul[D1, D2, D3]: DimMul[(D1, D2), (D2, D3), (D1, D3)] = neverRun

  // Shape Reverse
  trait ReverseLoop[A, Left, Acc]

  implicit def reverseBase[A, L <: Dimension, R](implicit l: ReverseLoop[A, L, R]): ShapeReverse[A, (R, L)] = neverRun

  implicit def reverseLoop[Init, L, A, Acc](implicit l: ReverseLoop[A, (Init, L), Acc]): ReverseLoop[A, Init, (Acc, L)] = neverRun

  implicit def reverseLoopStart[S, A]: ReverseLoop[(S, A), S, A] = neverRun

  // TNumber to int
  implicit val T0Value = TNumberValue[T0](0)

  implicit def TNValue[N <: TNumber](implicit v: TNumberValue[N]) = TNumberValue[TN[N]](v.value+1)

  // TNumber for indexing Shape

  implicit def inRangeBase2[D <: Dimension, S]: InRange[T0, (S, D), S] = neverRun

  implicit def inRangeTn1[D1 <: Dimension, D2 <: Dimension]: InRange[TNumber.t1, (D1, D2), D2] = neverRun

  implicit def inRangeRec2[D <: Dimension, N <: TNumber, S, S1, C](implicit i: InRange[N, S, S1]): InRange[TN[N], (S, D), (S1, D)] = neverRun

  // Shape Index as tuples
  implicit def indexVector[D<:Dimension]: ShapeToIndex[D, Int] = neverRun

  implicit def indexTensor[D<:Dimension,S,R1](implicit r1: ShapeToIndex[S,R1]): ShapeToIndex[(S,D),(R1,Int)] = neverRun
}
