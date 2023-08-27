namespace ControlnetApp.Domain
{
    public struct Size
    {
        public Size(int width, int height)
        {
            Width = width;
            Height = height;
        }

        public int Width { get; init; }
        public int Height { get; init; }
    }
}
