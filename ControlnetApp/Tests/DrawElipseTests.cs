using ControlnetApp;
using SixLabors.ImageSharp.Drawing.Processing;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace Tests
{
    public static class DrawElipseTests
    {
        public static void DrawTest()
        {
            using var image = new Image<Rgba32>(512, 512);
            image.Mutate(x => x.Fill(Color.Black));



            var p1 = new Vector2(0, 512 / 2);
            var p2 = new Vector2(512, 512 / 2);

            //var p1 = new Vector2(512 / 2, 0);
            //var p2 = new Vector2(512 / 2, 512);

            var points = new List<PointF>();

            ArcElipse.Ellipse2Poly(p1, p2, points);

            var c = Color.Red;
            var stickwidth = 4;
            image.Mutate(x => x.DrawPolygon(c, stickwidth, points.ToArray()));
            image.Mutate(x => x.DrawLines(Color.Green, stickwidth, p1, p2));


            image.SaveAsPng("draw_test.png");
        }
    }
}
