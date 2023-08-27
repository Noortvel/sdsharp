using ControlnetApp.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline.OnnxRunners
{
    internal class StableDiffusionCn
    {
        private readonly InferenceSession _session;

        public StableDiffusionCn(SessionOptions options)
        {
            _session = new(ModelsPaths.StableDiffusion, options);
        }

        public Tensor<float> Run(
            Tensor<float> encoderHiddenStates,
            Tensor<float> sample,
            float timeStep,
            IReadOnlyList<Tensor<float>> controlnetOutput)
        {
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<float>(new float[] { timeStep }, new int[] { 1 })),
            };

            for (int i = 0; i < controlnetOutput.Count - 1; i++)
            {
                var namedValue = NamedOnnxValue.CreateFromTensor(
                    $"down_block_{i}",
                    controlnetOutput[i]);

                inputs.Add(namedValue);
            }

            var midBlock = NamedOnnxValue.CreateFromTensor(
                "mid_block_additional_residual",
                controlnetOutput[controlnetOutput.Count - 1]);
            inputs.Add(midBlock);

            using var result = _session.Run(inputs); //out_sample
            //foreach (var r in result)
            //{
            //    Console.WriteLine($"SD (Name={r.Name}, Shape={r.AsTensor<float>().Dimensions.EnumerableToString()}");
            //}

            return result.Single().AsTensor<float>().Clone();
        }
    }
}
