namespace ControlnetApp
{
    public class ModelsPaths
    {

        public const string BasePath =
            //@"C:\Workspace\Projects\Stable-Diffusion-ONNX-FP16\model\anyv3-fp16-autoslicing-cn_openpose\"
            //@"C:\Workspace\Projects\Stable-Diffusion-ONNX-FP16\model\sd1_5-fp16-autoslicing-cn_openpose\"
            @"C:\Workspace\Projects\Stable-Diffusion-ONNX-FP16\model\deliberate_v2_cn_fp16\"
            ;

        public const string Controlnet =
            BasePath + @"controlnet\model.onnx";

        public const string StableDiffusion =
            BasePath + @"unet\model.onnx";

        public const string Tokenizer =
            BasePath + @"custom_tokenizer\custom_op_cliptok.onnx";

        public const string TextEncoder =
            BasePath + @"text_encoder\model.onnx";

        public const string VaeDecoder =
            BasePath + @"vae_decoder\model.onnx";

        public const string VaeEncoder =
            BasePath + @"vae_encoder\model.onnx";

    }
}
