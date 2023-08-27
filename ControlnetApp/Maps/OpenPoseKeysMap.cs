using ControlnetApp.Domain;

namespace ControlnetApp.Maps
{
    internal static class OpenPoseKeysMap
    {
        private static readonly IReadOnlyDictionary<int, OpenPoseKeys> Forward =
            new Dictionary<int, OpenPoseKeys>()
        {
                    {0, OpenPoseKeys.Nose},
                    {1, OpenPoseKeys.Neck},
                    {2, OpenPoseKeys.RSho},
                    {3, OpenPoseKeys.RElb},
                    {4, OpenPoseKeys.RWr},
                    {5, OpenPoseKeys.LSho},
                    {6, OpenPoseKeys.LElb},
                    {7, OpenPoseKeys.LWr},
                    {8, OpenPoseKeys.RHip},
                    {9, OpenPoseKeys.RKnee},
                    {10, OpenPoseKeys.RAnk},
                    {11, OpenPoseKeys.LHip},
                    {12, OpenPoseKeys.LKnee},
                    {13, OpenPoseKeys.LAnk},
                    {14, OpenPoseKeys.REye},
                    {15, OpenPoseKeys.LEye},
                    {16, OpenPoseKeys.REar},
                    {17, OpenPoseKeys.LEar},
        };

        public static readonly int Count = Forward.Count;

        private static readonly IReadOnlyDictionary<OpenPoseKeys, int> Inverse =
            Forward.ToDictionary(x => x.Value, x => x.Key);

        public static OpenPoseKeys MapIndex(int value) => Forward[value];

        public static int MapEnum(OpenPoseKeys value) => Inverse[value];
    }
}
