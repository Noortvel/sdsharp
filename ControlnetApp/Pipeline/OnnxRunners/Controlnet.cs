using ControlnetApp.Extensions;
using Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp.Pipeline.OnnxRunners
{
    internal class Controlnet
    {
        private readonly InferenceSession _session;

        //torch.logspace(-1, 0, 13)
        private readonly float[] GuidanceScales = new float[]
        {
            0.1000f, 0.1212f, 0.1468f, 0.1778f, 0.2154f,
            0.2610f, 0.3162f, 0.3831f, 0.4642f,
            0.5623f, 0.6813f, 0.8254f, 1.0000f
        };

        public Controlnet(SessionOptions options)
        {
            _session = new(ModelsPaths.Controlnet, options);
        }

        public IReadOnlyList<Tensor<float>> Run(
            Tensor<float> encoderHiddenStates,
            Tensor<float> sample,
            Tensor<float> controlnetСond,
            float timeStep,
            float controlnetCondScale,
            float controlnetUncondScale,
            bool isGuidance)
        {
            //foreach(var meta in _session.InputMetadata)
            //{
            //    var v = meta.Value;
            //    Console.WriteLine($"{meta.Key}, Type: {v.OnnxValueType}, Dim:{v.Dimensions.EnumerableToString()}, ElemType: {v.ElementDataType}");
            //}

            // Scaling in app after model.
            var condScaleT = new DenseTensor<double>(1);
            condScaleT[0] = 1;
            //condScaleT[0] = controlnetCondScale;

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("controlnet_cond", controlnetСond),
                NamedOnnxValue.CreateFromTensor(
                    "timestep",
                    new DenseTensor<float>(new float[] { timeStep }, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor(
                    "conditioning_scale",
                    condScaleT)
            };

            using var result = _session.Run(input);
            var outputTensors = result
                .Select(x => x.AsTensor<float>().Clone())
                .ToArray();

            if (isGuidance)
            {
                for (int i = 0; i < outputTensors.Length; i++)
                {
                    var scale = GuidanceScales[i] * controlnetCondScale;
                    outputTensors[i].Mult(scale);
                }

                //foreach (var outputTensor in outputTensors)
                //{
                //    outputTensor.Mult(controlnetUncondScale, 0);
                //    outputTensor.Mult(controlnetCondScale, 1);
                //}
            }
            else
            {
                foreach (var outputTensor in outputTensors)
                {
                    outputTensor.Mult(controlnetCondScale);
                }
            }


            return outputTensors;
        }
    }
}
