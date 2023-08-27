using Microsoft.ML.OnnxRuntime.Tensors;
using ControlnetApp.Pipeline.OnnxRunners;
using System.Diagnostics;
using ControlnetApp.Pipeline.Scheduler;
using Extensions;

namespace ControlnetApp.Pipeline
{
    /// <summary>
    /// Stable diffusion with controlnet pipeline.
    /// </summary>
    internal class StableDiffusionControlnetPipeline
    {
        private readonly TextPreprocessor _textPreprocessor;
        private readonly Controlnet _controlnet;
        private readonly StableDiffusionCn _stableDiffusionCn;
        private readonly VaeDecoder _vaeDecoder;

        public StableDiffusionControlnetPipeline(
            TextPreprocessor textPreprocessor,
            Controlnet controlnet,
            StableDiffusionCn stableDiffusionCn,
            VaeDecoder vaeDecoder)
        {
            _textPreprocessor = textPreprocessor;
            _controlnet = controlnet;
            _stableDiffusionCn = stableDiffusionCn;
            _vaeDecoder = vaeDecoder;
        }

        private DenseTensor<float>? SameLatents = null;

        public Image Inference(
            StableDiffusionConfig config,
            string prompt,
            Image<Rgba32> controlnetCond,
            string? negativePrompt = null,
            int? seed = null,
            bool isSameLatents = false)
        {
            bool isGuidance = config.GuidanceScale > 1;
            // Preprocess text
            var textEmbeddings = _textPreprocessor.Run(
                prompt,
                isGuidance,
                negativePrompt: negativePrompt);

            var scheduler = new EulerAncestralDiscreteScheduler();
            scheduler.SetTimesteps(config.NumInferenceSteps);
            var timestemps = scheduler.Timesteps;

            var currentSeed = seed ?? new Random().Next();
            var generator = new Random(currentSeed);

            DenseTensor<float> latents;
            if (isSameLatents)
            {
                if(SameLatents == null)
                {
                    SameLatents = GenerateLatentSamples(
                        1,
                        generator,
                        config,
                        scheduler.InitNoiseSigma);
                }

                latents = SameLatents;
            }
            else
            {
                latents = GenerateLatentSamples(
                1,
                generator,
                config,
                scheduler.InitNoiseSigma);
            }

            var controlNetTensor = PrepareCnImage(controlnetCond, isGuidance);

            DenseTensor<float> controlNetEmbeding = textEmbeddings;
            if(isGuidance)
            {
                controlNetEmbeding = textEmbeddings.GetSubDim0(1);
            }

            var stopWatch = new Stopwatch();
            for (int t = 0; t < timestemps.Count; t++)
            {
                var timestep = timestemps[t];


                var latentsInput =
                    isGuidance ?
                        latents.Dublicate2()
                    :
                        latents.Clone().AsDenseTensor();
                    ;

                scheduler.ScaleInput(
                        latentsInput.AsDenseTensor(),
                        timestep);

                stopWatch.Start();
                Tensor<float>[] controlnetOut;
                if (isGuidance)
                {
                    //Only positive
                    var controlnetLatents = latentsInput.GetSubDim0(1);
                    var positive = controlNetEmbeding;

                    var controlnetOutRaw = _controlnet.Run(
                        positive,
                        controlnetLatents,
                        controlNetTensor,
                        timestep,
                        config.ControlnetConditionScale,
                        config.ControlnetUnConditionScale,
                        isGuidance);
                    controlnetOut = new Tensor<float>[controlnetOutRaw.Count];

                    for(int i = 0; i < controlnetOut.Length; i++)
                    {
                        var val = TensorHelper.PrependZeros(controlnetOutRaw[i]);
                        controlnetOut[i] = val;
                    }
                }
                else
                {
                    var controlnetOutRaw = _controlnet.Run(
                        textEmbeddings,
                        latentsInput,
                        controlNetTensor,
                        timestep,
                        config.ControlnetConditionScale,
                        config.ControlnetUnConditionScale,
                        isGuidance);

                    controlnetOut = new Tensor<float>[controlnetOutRaw.Count];
                    for (int i = 0; i < controlnetOut.Length; i++)
                    {
                        var val = controlnetOutRaw[i];
                        controlnetOut[i] = val;
                    }
                }

                stopWatch.Stop();
                Console.WriteLine($"Controlnet time: {stopWatch.Elapsed}");
                stopWatch.Reset();

                stopWatch.Start();
                var stableDiffustionCnOut = _stableDiffusionCn.Run(
                    textEmbeddings,
                    latentsInput,
                    timestep,
                    controlnetOut);

                stopWatch.Stop();
                Console.WriteLine($"SD time: {stopWatch.Elapsed}");
                stopWatch.Reset();

                Tensor<float> noisePred = stableDiffustionCnOut;
                // Split tensors from 2,4,64,64 to 1,4,64,64 to Negative, Positve Prompts
                if (isGuidance)
                {
                    var (noisePredNegative, noisePredPositive) = TensorHelper.SplitTensor(
                        stableDiffustionCnOut,
                        2);
                    // Perform guidance, Diffuse Positive and negative promts predictions
                    noisePred = PerformGuidance(
                        noisePredNegative,
                        noisePredPositive,
                        config.GuidanceScale);
                }

                // Scheduler Step
                latents = scheduler.Step(noisePred, timestep, latents);
            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(
                latents.ToArray(),
                (1.0f / 0.18215f),
                latents.Dimensions.ToArray());

            stopWatch.Start();
            //int i = 0;
            //var vaeTensor = TensorHelper.CreateTensor(
            //    (latents as DenseTensor<float>)!.Buffer.Slice(i * latents.Strides[0], latents.Strides[0]).ToArray(), new[] { 1, 4, config.Height / 8, config.Width / 8 });
            var imageResultTensor = _vaeDecoder.Run(latents);
            stopWatch.Stop();
            Console.WriteLine($"VaeDecode time: {stopWatch.Elapsed}");
            stopWatch.Reset();

            var image = ConvertToImage(imageResultTensor, config.Width, config.Height);
            return image;
        }

        private static Tensor<float> PrepareCnImage(Image<Rgba32> image, bool isGuidacne)
        {
            var batch = 1; // isGuidacne ? 2 : 1;
            var tensor = new DenseTensor<float>(new[] { batch, 3, image.Height, image.Width});
            var scale = 1 / 255f;

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        var r = pixelSpan[x].R * scale;
                        var g = pixelSpan[x].G * scale;
                        var b = pixelSpan[x].B * scale;
                        for(int i = 0; i < batch; i++)
                        {
                            tensor[i, 0, y, x] = r;
                            tensor[i, 1, y, x] = g;
                            tensor[i, 2, y, x] = b;
                        }
                    }
                }
            });
            return tensor;
        }

        public static DenseTensor<float> GenerateLatentSamples(
            int batchSize,
            Random generator,
            StableDiffusionConfig config,
            float initNoiseSigma)
        {
            int numImagesPerPrompt = 1;
            int channels = 4;

            var latents = new DenseTensor<float>(
                new[] { batchSize * numImagesPerPrompt,
                    channels,
                    config.Height / 8,
                    config.Width / 8 });

            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = generator.NextDouble();                   // Uniform(0,1) random number
                var u2 = generator.NextDouble();                   // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1));       // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2;                    // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // Add noise to latents (scaled by scheduler.InitNoiseSigma)
                // Generate randoms that are negative and positive
                latentsArray[i] = (float)(standardNormalRand * initNoiseSigma);
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;
        }

        /// <summary>
        /// Returns dest.
        /// </summary>
        /// <param name="dest"></param>
        /// <param name="noisePredUncondition"></param>
        /// <param name="noisePredCondition"></param>
        /// <param name="guidanceScale"></param>
        /// <returns></returns>
        private static Tensor<float> PerformGuidance(
            Tensor<float> noisePredUncondition,
            Tensor<float> noisePredCondition,
            float guidanceScale)
        {
            Tensor<float> dest = noisePredCondition.CloneEmpty();
            for (int i = 0; i < noisePredUncondition.Length; i++)
            {
                var uncond = noisePredUncondition.GetValue(i);
                var cond = noisePredCondition.GetValue(i);
                //var newCond = cond + guidanceScale * (uncond - cond);
                var newCond = uncond + guidanceScale * (cond - uncond);
                dest.SetValue(i, newCond);
            }

            //noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);

            return dest;
        }

        private static Image<Rgb24> ConvertToImage(
            Tensor<float> output,
            int width,
            int height)
        {
            var result = new Image<Rgb24>(width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgb24(
                        (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            return result;
        }
    }
}
