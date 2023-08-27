using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;

namespace ControlnetApp.Extensions
{
    /// <summary>
    /// Math helper.
    /// </summary>
    public static class MH
    {
        public static float[] Linspace(
            float start,
            float end,
            int count)
        {
            var step = (end - start) / count;
            return Enumerable
                    .Range(0, count)
                    .Select(x => start + step * x)
                    .ToArray();
        }

        public static DenseTensor<float> LinspaceT(
            float start,
            float end,
            int count)
        {
            var step = (end - start) / count;
            var tensor = new DenseTensor<float>(count);
            for (int i = 0; i < count; i++)
            {
                tensor[i] = start + step * i;
            }

            return tensor;
        }

        public static DenseTensor<float> LinspaceSquaredT(
            float start,
            float end,
            int count)
        {
            var startS = MathF.Sqrt(start);
            var endS = MathF.Sqrt(end);
            var step = (endS - startS) / count;

            var tensor = new DenseTensor<float>(count);

            for (int i = 0; i < count; i++)
            {
                var value = startS + step * i;
                tensor[i] = value * value;
            }

            return tensor;
        }

        public static DenseTensor<float> Cumprod(
            Tensor<float> tensor)
        {
            var result = new DenseTensor<float>((int)tensor.Length);
            for (int i = 0; i < tensor.Length; i++)
            {
                float value = 1;
                for (int j = 0; j <= i; j++)
                {
                    value *= tensor[j];
                }

                result[i] = value;
            }

            return result;
        }
    }
}
