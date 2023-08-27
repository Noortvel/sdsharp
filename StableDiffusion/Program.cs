using StableDiffusion.ML.OnnxRuntime;
using System.Text.Encodings.Web;

namespace StableDiffusion
{
    public class Program
    {
        static void Main(string[] args)
        {
            //test how long this takes to execute
            var watch = System.Diagnostics.Stopwatch.StartNew();

            //Default args
            var prompt = "1girl, blonde, long dress, dancing, best quality";
            Console.WriteLine(prompt);

            var basePath = @"C:\Workspace\Projects\StableDiffusion\StableDiffusion\models\";
            var config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 0,
                // Update paths to your models
                TextEncoderOnnxPath = basePath + @"text_encoder\model.onnx",
                UnetOnnxPath = basePath + @"unet\model.onnx",
                VaeDecoderOnnxPath = basePath + @"vae_decoder\model.onnx",
                SafetyModelPath = basePath + @"safety_checker\model.onnx",
            };

            //sd_controlnet_openpose

            // Inference Stable Diff
            var image = UNet.Inference(prompt, config);

            // If image failed or was unsafe it will return null.
            if (image == null)
            {
                Console.WriteLine("Unable to create image, please try again.");
            }
            // Stop the timer
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine("Time taken: " + elapsedMs + "ms");

        }

    }
}
