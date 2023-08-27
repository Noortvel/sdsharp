using System.Numerics;

namespace ControlnetApp
{
    public static class ArcElipse
    {
        private const float Rad2Deg = 180f / MathF.PI;
        private const float Deg2Rad = MathF.PI / 180f;

        public static void Ellipse2Poly(
           Vector2 from,
           Vector2 to,
           List<PointF> points)
        {
            var dt = to - from;
            var dtLen = dt.Length();

            var pAngle = (int)(MathF.Atan2(dt.Y, dt.X) * Rad2Deg);
            var center = (from + to) / 2;

            var axeX = dtLen / 2;
            var axeY = 3;
            var axes = new Vector2(axeX, axeY);

            ArcElipse.Ellipse2Poly(
                center,
                axes,
                pAngle,
                0,
                360,
                1,
                points);
        }

        public static void Ellipse2Poly(
            Vector2 center,
            Vector2 axes,
            int angle,
            int arc_start,
            int arc_end,
            int delta,
            List<PointF> pts)
        {
            pts.Clear();

            float alpha, beta;
            int i;

            while (angle < 0)
                angle += 360;
            while (angle > 360)
                angle -= 360;

            if (arc_start > arc_end)
            {
                i = arc_start;
                arc_start = arc_end;
                arc_end = i;
            }
            while (arc_start < 0)
            {
                arc_start += 360;
                arc_end += 360;
            }
            while (arc_end > 360)
            {
                arc_end -= 360;
                arc_start -= 360;
            }
            if (arc_end - arc_start > 360)
            {
                arc_start = 0;
                arc_end = 360;
            }

            var (a_sin, a_cos) = MathF.SinCos(angle * Deg2Rad);

            for (i = arc_start; i < arc_end + delta; i += delta)
            {
                double x, y;
                var i_angle = i;
                if (i_angle > arc_end)
                    i_angle = arc_end;
                if (i_angle < 0)
                    i_angle += 360;

                x = axes.X * MathF.Cos(i_angle * Deg2Rad);
                y = axes.Y * MathF.Sin(i_angle * Deg2Rad);

                PointF pt = new();
                pt.X = center.X + (float)x * a_cos - (float)y * a_sin;
                pt.Y = center.Y + (float)x * a_sin + (float)y * a_cos;
                pts.Add(pt);
            }

            // If there are no points, it's a zero-size polygon
            if (pts.Count == 1)
            {
                pts.Add(center);
            }
        }
    }
}
