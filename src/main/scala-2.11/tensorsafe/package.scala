/**
 * Created by weijiayi on 10/10/2016.
 */
package object tensorsafe {

  type ~[A, B] = (A, B)

  implicit class Tilde[A](a: A){
    def ~[B](b: B) = (a,b)
  }

  val unitDim = DimValue.const[UnitDim](1)

  def scalar(x: Double): Tensor[UnitDim] = (TensorBuilder > unitDim).create(Array(x))

  def dim[D<:Dimension](d: Int) = DimValue.const[D](d)

  val tb = TensorBuilder

  private[tensorsafe] def cast[A](x: Any): A = x.asInstanceOf[A]

  def broadCastShape(s1: Array[Int], s2: Array[Int]): Array[Int] = {
    val s1Len = s1.length
    val s2Len = s2.length
    val newLen = math.max(s1Len, s2Len)
    val result = new Array[Int](newLen)
    for(i<-0 until newLen) {
      val i1 = s1Len - 1 - i
      val i2 = s2Len - 1 - i
      val d =
        if (i1 >= 0 && i2 >= 0) math.max(s1(i1), s2(i2))
        else if (i1 >= 0) s1(i1)
        else if (i2 >= 0) s2(i2)
        else throw new Exception(s"broadcast between ${s1.mkString(",")} and ${s2.mkString(",")} failed!")
      result(newLen-1-i) = d
    }
    result
  }
}
