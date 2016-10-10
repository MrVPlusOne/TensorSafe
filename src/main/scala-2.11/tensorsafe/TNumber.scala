package tensorsafe

/**
 * Created by weijiayi on 10/10/2016.
 */
sealed trait TNumber
sealed trait T0 extends TNumber
sealed trait TN[TN<:TNumber] extends TNumber

object TNumber{
  type t0 = T0
  type t1 = TN[t0]
  type t2 = TN[t1]
  type t3 = TN[t2]
  type t4 = TN[t3]
  type t5 = TN[t4]
  type t6 = TN[t5]
  type t7 = TN[t6]
  type t8 = TN[t7]
  type t9 = TN[t8]
  type t10 = TN[t9]
  type t11 = TN[t10]
  type t12 = TN[t11]
  type t13 = TN[t12]
  type t14 = TN[t13]
  type t15 = TN[t14]
  type t16 = TN[t15]
  type t17 = TN[t16]
  type t18 = TN[t17]
  type t19 = TN[t18]
  type t20 = TN[t19]
  type t21 = TN[t20]
  type t22 = TN[t21]
  type t23 = TN[t22]

  val n0 = new T0{}
  val n1 = new t1{}
}
