using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline.OnnxRunners
{
    internal class TextEncoder
    {
        private readonly InferenceSession _session;

        public TextEncoder(SessionOptions options)
        {
            _session = new(ModelsPaths.TextEncoder, options);
        }

        /// <summary>
        /// Run.
        /// </summary>
        /// <param name="inputIds">input_ids[77]</param>
        /// <returns>lastHiddenState[1x77x768] in flat represent</returns>
        public DisposableValue<Tensor<float>> Run(int[] inputIds)
        {
            var tensor = new DenseTensor<int>(inputIds, new int[] { 1, inputIds.Length });
            var namedInput = NamedOnnxValue.CreateFromTensor("input_ids", tensor);

            var result = _session.Run(new[] { namedInput });

            var lastHiddenState = result.First().AsTensor<float>();
            return new(lastHiddenState, result);
        }
    }
}
