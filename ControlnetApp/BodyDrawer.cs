using ControlnetApp.Domain;
using ControlnetApp.Extensions;
using ControlnetApp.Maps;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using System;
using System.Numerics;

namespace ControlnetApp
{
    internal static class BodyDrawer
    {
        /// <summary>
        /// Len 18.
        /// </summary>
        private static readonly IReadOnlyList<Rgb24> Colors = new List<Rgb24>()
        {
            new (255, 0, 0),
            new (255, 85, 0),
            new (255, 170, 0),
            new (255, 255, 0),
            new (170, 255, 0),
            new (85, 255, 0),
            new (0, 255, 0),
            new (0, 255, 85),
            new (0, 255, 170),
            new (0, 255, 255),
            new (0, 170, 255),
            new (0, 85, 255),
            new (0, 0, 255),
            new (85, 0, 255),
            new (170, 0, 255),
            new (255, 0, 255),
            new (255, 0, 170),
            new (255, 0, 85),
        };

        /// <summary>
        /// Len 19.
        /// </summary>
        private static readonly IReadOnlyList<(int From, int To)> LineIndexes = new List<(int From, int To)>()
        {
            (1,2),
            (1,5),
            (2,3),
            (3,4),
            (5,6),
            (6,7),
            (1,8),
            (8,9),
            (9,10),
            (1,11),
            (11,12),
            (12,13),
            (1,0),
            (0,14),
            (14,16),
            (0,15),
            (15,17),
            (2,17), //(2,17),
            (5,16),
        };

        const int KeysCount = 17;
        public static Image<Rgba32> DrawAutobox(
            SdSize sdSize,
            Domain.Size imageSize,
            IReadOnlyList<Vector2i> cocoPoints)
        {
            if(cocoPoints.Count != KeysCount)
            {
                throw new ArgumentException(
                    "Points count must be 17",
                    paramName: nameof(cocoPoints));
            }

            var points = CocoToOpenPoseMap.CocoToOpenpose(cocoPoints);

            var sdMinSide = MathF.Min(sdSize.Width, sdSize.Height);
            var sdMinSideMid = sdMinSide / 2;

            var image = new Image<Rgba32>(sdSize.Width, sdSize.Height, Color.Black);
            image.GetConfiguration().GetGraphicsOptions().Antialias = false;
            //image.Mutate(x => x.Fill(Color.Black));

            //var maxImgSize = MathF.Max(imageSize.Width, imageSize.Height);

            var minMax = CalcMinMax(points);
            var hbWidth = minMax.MinMaxX[1] - minMax.MinMaxX[0];
            var hbHeight = minMax.MinMaxY[1] - minMax.MinMaxY[0];
            var mmLen = MathF.Max(
                hbWidth,
                hbHeight);

            const float stickwidth = 4f;
            var elipsePoints = new List<PointF>();
            for (int i = 0; i < points.Count - 1; i++) // 17 Iter
            {
                //DRAW
                var line = LineIndexes[i];

                int i1 = line.From;
                int i2 = line.To;
                var p1 = points[i1];
                var p2 = points[i2];

                //var pNormed1 = Normalize(
                //    p1,
                //    new(hbWidth, hbHeight),
                //    minMax.MinMaxX[0],
                //    minMax.MinMaxY[0],
                //    sdMinSide,
                //    mmLen,
                //    sdMinSideMid
                //    );

                //var pNormed2 = Normalize(
                //    p2,
                //    new(hbWidth, hbHeight),
                //    minMax.MinMaxX[0],
                //    minMax.MinMaxY[0],
                //    sdMinSide,
                //    mmLen,
                //    sdMinSideMid
                //    );

                var pNormed1 = p1;
                var pNormed2 = p2;

                ArcElipse.Ellipse2Poly(pNormed1, pNormed2, elipsePoints);

                var c = Colors[i];
                var ca = AddDefaultTransparenty(c);
                image.Mutate(x => x.FillPolygon(ca, elipsePoints.ToArray()));
            }

            const float circleRadius = 3; //3
            for(int i = 0; i < points.Count; i++)
            {
                var c = Colors[i];
                //var ca = AddDefaultTransparenty(c);

                var p = points[i];
                //var np = Normalize(
                //    p,
                //    new(hbWidth, hbHeight),
                //    minMax.MinMaxX[0],
                //    minMax.MinMaxY[0],
                //    sdMinSide,
                //    mmLen,
                //    sdMinSideMid
                //    );
                var np = p;
                var elipse = new EllipsePolygon(ToPointF(np), circleRadius);
                image.Mutate(x => x.Fill(c, elipse));
            }

            return image;
        }

        private static Rgba32 AddDefaultTransparenty(Rgb24 c)
            => new Rgba32(c.R, c.G, c.B, 120);

        private static Vector2 Normalize(
            Vector2 value,
            Vector2 humanBox,
            float minX,
            float minY,
            float sdMinSide,
            float mmLen,
            float sdMinSideMid)
        {
            var pNormed = value;
            pNormed.X = pNormed.X - minX;
            pNormed.Y = pNormed.Y - minY;

            pNormed.X *= sdMinSide / mmLen;
            pNormed.Y *= sdMinSide / mmLen;

            pNormed.X += sdMinSideMid - humanBox[0] / 2;
            pNormed.Y += sdMinSideMid - humanBox[1] / 2;

            return pNormed;
        }

        private static (Vector2 MinMaxX, Vector2 MinMaxY) CalcMinMax(
            IReadOnlyList<Vector2> points)
        {
            var minX = float.MaxValue;
            var minY = float.MaxValue;
            var maxX = float.MinValue;
            var maxY = float.MinValue;

            for (int i = 0; i < points.Count; i++)
            {
                var p = points[i];

                if (p.X < minX)
                {
                    minX = p.X;
                }

                if (p.X > maxX)
                {
                    maxX = p.X;
                }

                if (p.Y < minY)
                {
                    minY = p.Y;
                }

                if (p.Y > maxY)
                {
                    maxY = p.Y;
                }
            }

            return (new(minX, maxX), new(minY, maxX));
        }

        private static PointF ToPointF(Vector2i point)
        {
            return new PointF(point.X, point.Y);
        }


        private static PointF ToPointF(Vector2 point)
        {
            return new PointF(point.X, point.Y);
        }

        private static PointF ToPointF(Vector<float> point)
        {
            return new PointF(point[0], point[1]);
        }


    }
}
