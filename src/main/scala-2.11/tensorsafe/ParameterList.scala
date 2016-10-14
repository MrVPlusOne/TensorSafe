package tensorsafe

import tensorsafe.TensorOps.ShapeValue

/**
 * Created by weijiayi on 11/10/2016.
 */
object ParameterList {
  def tensorsToParams[TS](tensors: TS)(implicit tensorList: TensorList[TS]): IndexedSeq[Double] = tensorList.flatten(tensors)

  def paramsToTensors[TS](flattened: IndexedSeq[Double])(implicit tensorList: TensorList[TS]): TS = tensorList.reconstruct(flattened)


  trait TensorList[TS]{
    def flatten(tensors: TS): IndexedSeq[Double]
    def reconstruct(flattened: IndexedSeq[Double]): TS
  }

  object TensorList{
    // Tensor List
    implicit val tensorListBasic: TensorList[RNil] = new TensorList[RNil] {
      override def flatten(tensors: RNil): IndexedSeq[Double] = IndexedSeq()

      override def reconstruct(flattened: IndexedSeq[Double]): RNil = RNil
    }

    implicit def tensorListRec[S,TS](implicit ts: TensorList[TS], sv: ShapeValue[S]): TensorList[TS~Tensor[S]] = new TensorList[TS~Tensor[S]] {
      override def flatten(tensors: (TS, Tensor[S])): IndexedSeq[Double] =  tensors._2.data ++ ts.flatten(tensors._1)

      override def reconstruct(flattened: IndexedSeq[Double]): (TS, Tensor[S]) = {
        val size = sv.shape.product
        val (l,r) = flattened.splitAt(size)
        ts.reconstruct(r) ~ Tensor.create[S](l.toArray)
      }
    }
  }

}
