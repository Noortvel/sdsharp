using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;
using static System.Collections.Specialized.BitVector32;

namespace Extensions
{
    internal static class TensorExtensions
    {
        public static void Mult(this Tensor<float> tensor, float val)
        {
            var dense = tensor.AsDenseTensor();
            var vector = dense.AsVector();
            var result = vector *= val;
            result.CopyTo(dense.Buffer.Span);

            //for (int i = 0; i < tensor.Length; i++)
            //{
            //    var el = tensor.GetValue(i);
            //    tensor.SetValue(i, el * val);
            //}
        }

        public static void Mult(this Tensor<float> tensor, float val, int dim0)
        {
            int sliceLen = 1;
            for(int i = 1; i < tensor.Dimensions.Length; i++)
            {
                sliceLen *= tensor.Dimensions[i];
            }

            for (int i = 0; i < sliceLen; i++)
            {
                var t_index = sliceLen * dim0 + i;
                var el = tensor.GetValue(t_index);
                tensor.SetValue(t_index, el * val);
            }
        }

        public static Vector<float> ToVector(this Tensor<float> tensor)
        {
            var v = new Vector<float>(tensor.ToArray());
            return v;
        }

        public static Vector<float> AsVector(this DenseTensor<float> tensor)
        {
            var v = new Vector<float>(tensor.Buffer.Span);
            return v;
        }

        public static DenseTensor<float> GetSubDim0(
            this Tensor<float> tensorToSplit,
            int dim0)
        {
            var slicedStrides = 1;

            var newDimensions = tensorToSplit
                .Dimensions
                .ToArray();

            newDimensions[0] = 1;

            var lenght = 1;
            for(int i = 1; i < newDimensions.Length; i++)
            {
                lenght *= newDimensions[i];
            }
            var dense = (tensorToSplit as DenseTensor<float>)!;
            var resultTensor = new DenseTensor<float>(
                dense.Buffer.Slice(dim0 * lenght),
                newDimensions);

            return resultTensor;
        }

        public static DenseTensor<float> AsDenseTensor(this Tensor<float> tensor)
        {
            return (tensor as DenseTensor<float>) ??
                throw new ArgumentException("Is not dense tensor");
        }

        /// <summary>
        /// Copy [1, ...] tensor to [2, ....] tensor
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static DenseTensor<float> Dublicate2(this DenseTensor<float> tensor)
        {
            var dims = tensor.Dimensions.ToArray();
            dims[0] += 1;

            var newTensor = new DenseTensor<float>(dims);
            tensor.Buffer.Span.CopyTo(newTensor.Buffer.Span.Slice(0));
            tensor.Buffer.Span.CopyTo(newTensor.Buffer.Span.Slice((int)tensor.Length));

            return newTensor;
        }
    }
}
