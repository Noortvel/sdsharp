using ControlnetApp.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline.OnnxRunners
{
    internal class VaeDecoder
    {
        private readonly InferenceSession _session;

        public VaeDecoder(SessionOptions options)
        {
            _session = new(ModelsPaths.VaeDecoder, options);
        }

        public Tensor<float> Run(Tensor<float> latents)
        {
            //foreach (var meta in _session.InputMetadata)
            //{
            //    var v = meta.Value;
            //    Console.WriteLine($"{meta.Key}, Type: {v.OnnxValueType}, Dim:{v.Dimensions.EnumerableToString()}, ElemType: {v.ElementDataType}");
            //}

            var decoderInput = new NamedOnnxValue[] 
            { 
                NamedOnnxValue.CreateFromTensor("latent_sample", latents)
            };

            using var result = _session.Run(decoderInput);
            var output = result.Single().AsTensor<float>().Clone();
            return output;
        }
    }
}
