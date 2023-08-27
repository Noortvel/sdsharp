using Microsoft.ML.OnnxRuntime.Tensors;

namespace ControlnetApp
{
    internal static class InputControlnetCrutch
    {
        private static Image<Rgba32> _image;

        static InputControlnetCrutch()
        {
            _image = Image.Load<Rgba32>("openpose-pose (1).png");
            //_image = Image.Load<Rgba32>("crutch_pose1.png");
            //_image = Image.Load<Rgb24>("control_human_openpose.png");
        }

        public static Image<Rgba32> Get()
        {
            return _image;
        }
    }
}
