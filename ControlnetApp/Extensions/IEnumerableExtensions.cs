namespace ControlnetApp.Extensions
{
    internal static class IEnumerableExtensions
    {
        private const string Sep = ", ";
        public static string EnumerableToString<T>(this IEnumerable<T> source)
            => $"[{string.Join(Sep, source.Select(x => x?.ToString()))}]";

        public static string EnumerableToString<T>(this ReadOnlySpan<T> source)
        {
            return source.ToArray().EnumerableToString();
        }
    }
}
