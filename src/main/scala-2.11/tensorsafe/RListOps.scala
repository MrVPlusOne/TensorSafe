package tensorsafe

import RListOps._

import scala.language.implicitConversions

class RListOps[L](l:L){
  def reverse(implicit rev: Reverse[L]): rev.Out = rev(l)
}

object RListOps {
  implicit def tuple2RList[A,B](t: (A,B)): RListOps[(A,B)] = new RListOps(t)

  trait Reverse[L]{
    type Out
    def apply(l: L): Out
  }

  object Reverse {
    type Aux[A, B] = Reverse[A]{ type Out = B}

    implicit def reverse[A, Out0](implicit rev0: ReverseL[A, RNil, Out0]): Aux[A, Out0] =
      new Reverse[A]{
        type Out = Out0
        override def apply(l: A): Out0 = rev0(l, RNil)
      }

    trait ReverseL[A, B, Out]{
      def apply(a: A, b: B): Out
    }

    object ReverseL{
      implicit def nilRnv[A]: ReverseL[RNil, A, A] = new ReverseL[RNil, A, A] {
        override def apply(a: RNil, b: A): A = b
      }

      implicit def recursion[A, B, C, Out0](implicit rev: ReverseL[A, B~C, Out0]): ReverseL[A~C, B, Out0]
      = new ReverseL[A~C, B, Out0]{
        override def apply(a: ~[A, C], b: B): Out0 = rev(a._1, b~a._2)
      }
    }

    object tests{
      val l = RNil ~ "a" ~ 53 ~ 1.5
      val lr1 = RNil ~ 1.5 ~ 53 ~ "a"
      val lr2 = l.reverse
    }
  }













}
