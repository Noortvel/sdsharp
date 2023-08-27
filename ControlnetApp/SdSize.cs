namespace ControlnetApp
{
    internal struct SdSize
    {
        public SdSize(int width, int height)
        {
            ThrowIfNot8Divided(width, nameof(width));
            ThrowIfNot8Divided(height, nameof(height));
            if(width != height)
            {
                throw new ArgumentException("Must be height == width");
            }

            Width = width;
            Height = height;
        }

        public int Width { get; }
        public int Height { get; }

        private static void ThrowIfNot8Divided(int value, string name)
        {
            if(value % 8 != 0)
            {
                throw new ArgumentException(
                    "Value must be devided on 8",
                    paramName: name);
            }
        }
    }
}
