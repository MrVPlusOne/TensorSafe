package tensorsafe

/**
 * Created by weijiayi on 11/10/2016.
 */
object ParameterList {
  trait TensorList[TS]{
    def flatten(tensors: TS): IndexedSeq[Double]
    def reconstruct(flattened: IndexedSeq[Double]): TS
  }

  def tensorsToParams[TS](tensors: TS)(implicit tensorList: TensorList[TS]): IndexedSeq[Double] = tensorList.flatten(tensors)

  def paramsToTensors[TS](flattened: IndexedSeq[Double])(implicit tensorList: TensorList[TS]): TS = tensorList.reconstruct(flattened)
}
