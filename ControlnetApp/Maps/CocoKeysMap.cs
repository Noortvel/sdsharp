using ControlnetApp.Domain;

namespace ControlnetApp.Maps
{
    internal static class CocoKeysMap
    {
        private static readonly IReadOnlyDictionary<int, CocoMocapKeys> Forward =
            new Dictionary<int, CocoMocapKeys>()
        {
                    {0, CocoMocapKeys.Nose},
                    {1, CocoMocapKeys.LeftEye},
                    {2, CocoMocapKeys.RightEye},
                    {3, CocoMocapKeys.LeftEar},
                    {4, CocoMocapKeys.RightEar},
                    {5, CocoMocapKeys.LeftShoulder},
                    {6, CocoMocapKeys.RightShoulder},
                    {7, CocoMocapKeys.LeftElbow},
                    {8, CocoMocapKeys.RightElbow},
                    {9, CocoMocapKeys.LeftWrist},
                    {10, CocoMocapKeys.RightWrist},
                    {11, CocoMocapKeys.LeftHip},
                    {12, CocoMocapKeys.RightHip},
                    {13, CocoMocapKeys.LeftKnee},
                    {14, CocoMocapKeys.RightKnee},
                    {15, CocoMocapKeys.LeftAnkle},
                    {16, CocoMocapKeys.RightAnkle}
        };

        public static readonly int Count = Forward.Count;

        private static readonly IReadOnlyDictionary<CocoMocapKeys, int> Inverse =
            Forward.ToDictionary(x => x.Value, x => x.Key);

        public static CocoMocapKeys MapIndex(int value) => Forward[value];

        public static int MapEnum(CocoMocapKeys value) => Inverse[value];
    }
}
