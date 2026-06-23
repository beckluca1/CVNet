using System.Numerics;

namespace CVNet;

class CVDetection
{
    private readonly byte Background = 0;
    private readonly byte Foreground = 255;
    private readonly byte VisitedOuterRight = 100;
    private readonly byte Visited = 200;

    private static readonly int[] dirX = { 1, 1, 0, -1, -1, -1, 0, 1 };
    private static readonly int[] dirY = { 0, -1, -1, -1, 0, 1, 1, 1 };

    private bool TraceContour(
        CVImage image,
        byte[] imageState,
        int startX,
        int startY,
        bool external,
        ref List<(int, int)> contour)
    {
        int height = image.Height;
        int width = image.Width;

        int x = startX;
        int y = startY;
        int index = x + width * y;

        int searchIdx = external ? 5 : 1;

        int startDir = -1;
        bool firstMove = true;

        while (true)
        {
            contour.Add((x, y));

            int foundDir = -1;

            for (int n = 0; n < 8; n++)
            {
                int dir = (searchIdx + n) & 7;

                // Get test location
                int nx = x + dirX[dir];
                int ny = y + dirY[dir];
                int nextIndex = nx + width * ny;

                // If location is out of bounds skip
                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                    continue;

                // If test if background (not foreground) skip
                if (imageState[nextIndex] == Background)
                    continue;

                // Update foundDir
                foundDir = dir;

                if (((searchIdx <= 1) || (dir <= searchIdx - 2)) && (x != startX && y != startY))
                {
                    imageState[index] = VisitedOuterRight;
                }
                else if (imageState[index] == Foreground)
                {
                    imageState[index] = Visited;
                }

                x = nx;
                y = ny;
                index = nextIndex;

                searchIdx = (dir + 6) & 7;

                break;
            }

            if (firstMove)
            {
                if (foundDir == -1)
                {
                    imageState[index] = VisitedOuterRight;
                    return true;
                }

                startDir = foundDir;
                firstMove = false;
            }

            // Jacob stopping criterion
            if (x == startX &&
                y == startY &&
                foundDir == startDir)
            {
                return true;
            }
        }
    }

    public List<List<(int, int)>> FindContours<T1, T2>(CVImage image, T2 foreground) where T1 : struct, IEqualityOperators<T1, T1, bool> where T2 : struct
    {
        Span<T1> buffer = image.BufferAs<T1>();

        T1 fg = (T1)Convert.ChangeType(foreground, typeof(T1));

        int height = image.Height;
        int width = image.Width;

        byte[] imageState = new byte[width * height];

        for (int i = 0; i < height * width; i++) imageState[i] = (buffer[i] == fg) ? (byte)255 : (byte)0;

        var contours = new List<List<(int, int)>>();

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = x + width * y;

                //
                // External contour
                //
                if (imageState[index] == Foreground)
                {
                    var contour = new List<(int, int)>();

                    if (TraceContour(
                        image,
                        imageState,
                        x,
                        y,
                        true,
                        ref contour))
                    {
                        if (contour.Count > 1 &&
                            contour[0].Item1 == contour[^1].Item1 &&
                            contour[0].Item2 == contour[^1].Item2)
                        {
                            contour.RemoveAt(contour.Count - 1);
                        }

                        contours.Add(contour);
                    }
                }

                //
                // Internal contour
                //
                if (x > 0 &&
                    imageState[index - 1] > VisitedOuterRight)
                {
                    var contour = new List<(int, int)>();

                    if (TraceContour(
                        image,
                        imageState,
                        x - 1,
                        y,
                        false,
                        ref contour))
                    {
                        if (contour.Count > 1 &&
                            contour[0].Item1 == contour[^1].Item1 &&
                            contour[0].Item2 == contour[^1].Item2)
                        {
                            contour.RemoveAt(contour.Count - 1);
                        }

                        contours.Add(contour);
                    }
                }
            }
        }

        return contours;
    }

    public static void FindContours<T1, T2>(
        CVImage image,
        T2 foreground,
        ref List<List<(int, int)>> contours)
        where T1 : struct, IEqualityOperators<T1, T1, bool>
        where T2 : struct
    {
        Span<T1> buffer = image.BufferAs<T1>();

        int width = image.Width;
        int height = image.Height;
        int size = width * height;

        bool[] visited = new bool[size];

        T1 fg = (T1)Convert.ChangeType(foreground, typeof(T1));

        for (int y = 1; y < height - 1; y++)
        {
            int row = y * width;

            for (int x = 1; x < width - 1; x++)
            {
                int idx = row + x;

                if (visited[idx])
                    continue;

                if (buffer[image.SpanIndex(x, y, 0)] != fg)
                    continue;

                // fast inline border check (no function call)
                bool isBorder = false;

                for (int i = 0; i < 8; i++)
                {
                    int nx = x + dirX[i];
                    int ny = y + dirY[i];

                    int nidx = nx + ny * width;

                    if ((uint)nx >= width || (uint)ny >= height || buffer[nidx].Equals(default(T1)) || !buffer[nidx].Equals(fg))
                    {
                        isBorder = true;
                        break;
                    }
                }

                if (!isBorder)
                    continue;

                var contour = TraceContour(
                    image,
                    buffer,
                    x,
                    y,
                    fg,
                    visited,
                    width,
                    height);

                if (contour.Count > 2)
                    contours.Add(contour);
            }
        }
    }

    private static List<(int, int)> TraceContour<T>(
        CVImage image,
        Span<T> buffer,
        int startX,
        int startY,
        T fg,
        bool[] visited,
        int width,
        int height)
        where T : struct, IEqualityOperators<T, T, bool>
    {
        var contour = new List<(int, int)>(64);

        int x = startX;
        int y = startY;

        int sx = startX;
        int sy = startY;

        int dir = 0;

        do
        {
            contour.Add((x, y));

            int idx = x + y * width;
            visited[idx] = true;

            int startDir = dir;
            bool found = false;

            for (int i = 0; i < 8; i++)
            {
                int ndir = (startDir + i) & 7;

                int nx = x + dirX[ndir];
                int ny = y + dirY[ndir];

                if ((uint)nx >= width || (uint)ny >= height)
                    continue;

                int nidx = nx + ny * width;

                if (!buffer[nidx].Equals(fg))
                    continue;

                x = nx;
                y = ny;

                dir = (ndir + 5) & 7; // Suzuki–Abe backtracking rule
                found = true;
                break;
            }

            if (!found)
                break;

        } while (x != sx || y != sy);

        return contour;
    }

    public static List<List<(int, int)>> FindContours<T>(CVImage image, T foreground) where T : struct
    {
        List<List<(int, int)>> contours = new List<List<(int, int)>>();

        if (image.DataFormat == CVDataFormat.CV_U8) CVContour.TraceContours<byte, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S8) CVContour.TraceContours<sbyte, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U16) CVContour.TraceContours<ushort, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S16) CVContour.TraceContours<short, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U32) CVContour.TraceContours<uint, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S32) CVContour.TraceContours<int, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U64) CVContour.TraceContours<ulong, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S64) CVContour.TraceContours<long, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_F32) CVContour.TraceContours<float, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_F64) CVContour.TraceContours<double, T>(image, foreground, ref contours);

        Console.WriteLine($"Found contours {contours.Count}");

        return contours;
    }

    private static double Perimeter(List<(int x, int y)> contour)
    {
        double sum = 0.0;

        if (contour.Count == 0) return sum;

        (int, int) lastI = contour[0];

        for (int i = 1; i < contour.Count; i++)
        {
            (int, int) currentI = contour[i];
            int dX = (currentI.Item1 - lastI.Item1);
            int dY = (currentI.Item2 - lastI.Item2);
            sum += Math.Sqrt(dX * dX + dY * dY);
            lastI = currentI;
        }

        (int, int) currentEI = contour[0];
        int dEX = (currentEI.Item1 - lastI.Item1);
        int dEY = (currentEI.Item2 - lastI.Item2);
        sum += Math.Sqrt(dEX * dEX + dEY * dEY);

        return sum;
    }

    private static List<(int, int)> DouglasPeucker(
           List<(int x, int y)> points,
           double epsilon)
    {
        if (points.Count < 3)
            return new List<(int, int)>(points);

        int first = 0;
        int last = points.Count - 1;

        double maxDistance = 0;
        int index = 0;

        for (int i = first + 1; i < last; i++)
        {
            double distance = PerpendicularDistance(points[i], points[first], points[last]);

            if (distance > maxDistance)
            {
                maxDistance = distance;
                index = i;
            }
        }

        // If max distance is greater than epsilon, recursively simplify
        if (maxDistance > epsilon)
        {
            var left = DouglasPeucker(points.GetRange(first, index - first + 1), epsilon);
            var right = DouglasPeucker(points.GetRange(index, last - index + 1), epsilon);

            // merge results (avoid duplicate middle point)
            left.RemoveAt(left.Count - 1);
            left.AddRange(right);

            return left;
        }
        else
        {
            return new List<(int, int)>
            {
                points[first],
                points[last]
            };
        }
    }

    // ---------------- Distance from point to line ----------------

    private static double PerpendicularDistance(
        (int x, int y) p,
        (int x, int y) a,
        (int x, int y) b)
    {
        double ax = b.x - a.x;
        double ay = b.y - a.y;

        if (ax == 0 && ay == 0)
        {
            return Math.Sqrt(Math.Pow(p.x - a.x, 2) + Math.Pow(p.y - a.y, 2));
        }

        double t = ((p.x - a.x) * ax + (p.y - a.y) * ay) /
                   (ax * ax + ay * ay);

        t = Math.Max(0, Math.Min(1, t));

        double projX = a.x + t * ax;
        double projY = a.y + t * ay;

        return Math.Sqrt(Math.Pow(p.x - projX, 2) + Math.Pow(p.y - projY, 2));
    }

    public static double distance((int, int) a, (int, int) b)
    {
        int dX = b.Item1 - a.Item1;
        int dY = b.Item2 - a.Item2;
        return Math.Sqrt(dX * dX + dY * dY);
    }

    public static List<(int, int)> ReorderToLongest(List<(int, int)> contour)
    {
        int bestIndex = 0;
        double length = distance(contour[0], contour[contour.Count - 1]);

        for (int i = 1; i < contour.Count; i++)
        {
            double newLength = distance(contour[i], contour[i - 1]);

            if (newLength > length)
            {
                length = newLength;
                bestIndex = i;
            }
        }

        List<(int, int)> bestContour = new List<(int, int)>();

        for (int i = bestIndex; i < contour.Count; i++)
            bestContour.Add(contour[i]);
        for (int i = 0; i < bestIndex; i++)
            bestContour.Add(contour[i]);

        return bestContour;
    }


    public static List<List<(int, int)>> ApproximatePolygons(List<List<(int, int)>> contours, double epsilon)
    {
        List<List<(int, int)>> polygons = new List<List<(int, int)>>();

        foreach (var contour in contours)
        {
            double perimeter = Perimeter(contour);

            var approx = DouglasPeucker(contour, epsilon * perimeter);
            polygons.Add(approx);
        }

        return polygons;
    }

    private static double BoundingArea(List<(int, int)> quad)
    {
        int minX = 1000;
        int minY = 1000;
        int maxX = 0;
        int maxY = 0;

        for (int i = 0; i < quad.Count; i++)
        {
            if (quad[i].Item1 < minX) minX = quad[i].Item1;
            if (quad[i].Item1 > maxX) maxX = quad[i].Item1;

            if (quad[i].Item2 < minY) minY = quad[i].Item2;
            if (quad[i].Item2 > maxY) maxY = quad[i].Item2;

        }

        return (maxX - minX) * (maxY - minY);
    }

    private static double QuadArea(List<(int, int)> quad)
    {
        double sum = 0;

        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            sum += quad[i].Item1 * quad[j].Item2;
            sum -= quad[i].Item2 * quad[j].Item1;
        }

        return Math.Abs(sum) * 0.5;
    }

    private static double QuadAspect(List<(int, int)> quad)
    {
        double a = distance(quad[0], quad[1]);
        double b = distance(quad[1], quad[2]);
        double c = distance(quad[2], quad[3]);
        double d = distance(quad[3], quad[0]);

        return (a + c) / (b + d);
    }

    static bool IsConvexQuad(List<(int, int)> quad)
    {
        int sign = 0;

        for (int i = 0; i < 4; i++)
        {
            var p0 = quad[i];
            var p1 = quad[(i + 1) % 4];
            var p2 = quad[(i + 2) % 4];

            long cross =
                (long)(p1.Item1 - p0.Item1) * (p2.Item2 - p1.Item2) -
                (long)(p1.Item2 - p0.Item2) * (p2.Item1 - p1.Item1);

            if (cross == 0) return false;

            int currentSign = Math.Sign(cross);

            if (sign == 0)
                sign = currentSign;
            else if (currentSign != sign)
                return false;
        }

        return true;
    }

    public static List<List<(int, int)>> QuadrilateralChecks(List<List<(int, int)>> contours, int width, int height)
    {
        List<List<(int, int)>> quadrilaterals = new List<List<(int, int)>>();

        int imageArea = width * height;

        foreach (var contour in contours)
        {
            if (contour.Count != 5) continue;

            double area = QuadArea(contour);
            if (area < 0.0005 * imageArea) continue;

            if (area > 0.8 * imageArea)
            {
                Console.WriteLine("Fails area Test");
                continue;
            }

            double aspect = QuadAspect(contour);
            if (aspect < 0.8 || aspect > 1.2) continue;
            //Console.WriteLine("Fails aspect Test");

            if (!IsConvexQuad(contour)) continue;
            //Console.WriteLine("Fails convex Test");

            quadrilaterals.Add(contour);
        }

        Console.WriteLine($"Found Quadrilaterals {quadrilaterals.Count}");

        return quadrilaterals;
    }
}