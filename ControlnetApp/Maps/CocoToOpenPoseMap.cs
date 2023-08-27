using ControlnetApp.Domain;
using ControlnetApp.Extensions;
using System.Numerics;

namespace ControlnetApp.Maps
{
    internal static class CocoToOpenPoseMap
    {
        private static readonly IReadOnlyList<(OpenPoseKeys, CocoMocapKeys)> SimpleMap =
            new (OpenPoseKeys, CocoMocapKeys)[]
            {
                (OpenPoseKeys.Nose, CocoMocapKeys.Nose),

                (OpenPoseKeys.LEye, CocoMocapKeys.LeftEye),
                (OpenPoseKeys.REye, CocoMocapKeys.RightEye),

                (OpenPoseKeys.LEar, CocoMocapKeys.LeftEar),
                (OpenPoseKeys.REar, CocoMocapKeys.RightEar),

                (OpenPoseKeys.LSho, CocoMocapKeys.LeftShoulder),
                (OpenPoseKeys.RSho, CocoMocapKeys.RightShoulder),

                (OpenPoseKeys.LElb, CocoMocapKeys.LeftElbow),
                (OpenPoseKeys.RElb, CocoMocapKeys.RightElbow),

                (OpenPoseKeys.LWr, CocoMocapKeys.LeftWrist),
                (OpenPoseKeys.RWr, CocoMocapKeys.RightWrist),

                (OpenPoseKeys.LHip, CocoMocapKeys.LeftHip),
                (OpenPoseKeys.RHip, CocoMocapKeys.RightHip),

                (OpenPoseKeys.LKnee, CocoMocapKeys.LeftKnee),
                (OpenPoseKeys.RKnee, CocoMocapKeys.RightKnee),

                (OpenPoseKeys.LAnk, CocoMocapKeys.LeftAnkle),
                (OpenPoseKeys.RAnk, CocoMocapKeys.RightAnkle),
            };

        public static IReadOnlyList<Vector2> CocoToOpenpose(IReadOnlyList<Vector2i> points)
        {
            var outPoints = new Vector2[OpenPoseKeysMap.Count];
            foreach (var m in SimpleMap)
            {
                outPoints[OpenPoseKeysMap.MapEnum(m.Item1)]
                    = points[CocoKeysMap.MapEnum(m.Item2)].ToVector2();
            }

            var ls = points[CocoKeysMap.MapEnum(CocoMocapKeys.LeftShoulder)].ToVector2();
            var rs = points[CocoKeysMap.MapEnum(CocoMocapKeys.RightShoulder)].ToVector2();

            outPoints[OpenPoseKeysMap.MapEnum(OpenPoseKeys.Neck)]
                = (ls + rs) / 2;

            return outPoints;
        }
    }
}
