using ControlnetApp.Domain;
using System.Numerics;

namespace ControlnetApp.Extensions
{
    public static class Vector2iExtensions
    {
        public static Vector2 ToVector2(this Vector2i vector)
            => new Vector2(vector.X, vector.Y);
    }
}
