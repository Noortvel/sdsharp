using ControlnetApp.Domain;
using System.Text.Json;

namespace ControlnetApp
{
    internal static class MocapResultRepository
    {
        private static readonly MocapResult _mocapResult;
        static MocapResultRepository()
        {
            var path = @"test_video_keypoints2.json";
            //var path = @"test_video_keypoints.json";
            var json = File.ReadAllText(path);
            var result = JsonSerializer.Deserialize<MocapResult>(json, new JsonSerializerOptions()
            {
                PropertyNameCaseInsensitive = true,
            });
            if (result == null)
            {
                throw new InvalidOperationException("");
            }

            _mocapResult = result;
        }

        public static MocapResult GetMocapResult()
        {
            return _mocapResult;
        }
    }
}
