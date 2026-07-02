using System.Drawing;
using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;


public class CVDetection
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
        ref CVContour contour)
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
            contour.points.Add((x, y));

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

    public List<CVContour> FindContours<T1, T2>(CVImage image, T2 foreground) where T1 : struct, IEqualityOperators<T1, T1, bool> where T2 : struct
    {
        Span<T1> buffer = image.BufferAs<T1>();

        T1 fg = (T1)Convert.ChangeType(foreground, typeof(T1));

        int height = image.Height;
        int width = image.Width;

        byte[] imageState = new byte[width * height];

        for (int i = 0; i < height * width; i++) imageState[i] = (buffer[i] == fg) ? (byte)255 : (byte)0;

        var contours = new List<CVContour>();

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
                    var contour = new CVContour();

                    if (TraceContour(
                        image,
                        imageState,
                        x,
                        y,
                        true,
                        ref contour))
                    {
                        if (contour.points.Count > 1 &&
                            contour.points[0].Item1 == contour.points[^1].Item1 &&
                            contour.points[0].Item2 == contour.points[^1].Item2)
                        {
                            contour.points.RemoveAt(contour.points.Count - 1);
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
                    var contour = new CVContour();

                    if (TraceContour(
                        image,
                        imageState,
                        x - 1,
                        y,
                        false,
                        ref contour))
                    {
                        if (contour.points.Count > 1 &&
                            contour.points[0].Item1 == contour.points[^1].Item1 &&
                            contour.points[0].Item2 == contour.points[^1].Item2)
                        {
                            contour.points.RemoveAt(contour.points.Count - 1);
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
        ref List<CVContour> contours)
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

                if (buffer[x + y * image.Width] != fg)
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

                if (contour.points.Count > 2)
                {
                    contours.Add(contour);
                }
            }
        }
    }

    private static CVContour TraceContour<T>(
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
        var contour = new CVContour();

        int x = startX;
        int y = startY;

        int sx = startX;
        int sy = startY;

        int dir = 0;

        do
        {
            contour.points.Add((x, y));

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

    public static List<CVContour> FindContours<T>(CVImage image, T foreground) where T : struct
    {
        List<CVContour> contours = new List<CVContour>();

        if (image.DataFormat == CVDataFormat.CV_U8) CVContourTrace.TraceContours<byte, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S8) CVContourTrace.TraceContours<sbyte, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U16) CVContourTrace.TraceContours<ushort, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S16) CVContourTrace.TraceContours<short, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U32) CVContourTrace.TraceContours<uint, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S32) CVContourTrace.TraceContours<int, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_U64) CVContourTrace.TraceContours<ulong, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_S64) CVContourTrace.TraceContours<long, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_F32) CVContourTrace.TraceContours<float, T>(image, foreground, ref contours);
        else if (image.DataFormat == CVDataFormat.CV_F64) CVContourTrace.TraceContours<double, T>(image, foreground, ref contours);

        return contours;
    }

    private static List<(int, int)> DouglasPeucker(
           List<(int, int)> points,
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

    public static double distanceSquared((int, int) a, (int, int) b)
    {
        int dX = b.Item1 - a.Item1;
        int dY = b.Item2 - a.Item2;
        return dX * dX + dY * dY;
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


    public static List<CVContour> ApproximatePolygons(List<CVContour> contours, double epsilon)
    {
        List<CVContour> polygons = new List<CVContour>();

        foreach (var contour in contours)
        {
            contour.setPerimeter();
            var approx = DouglasPeucker(contour.points, epsilon * contour.perimeter);
            polygons.Add(new CVContour(approx));
        }

        return polygons;
    }

    private static double QuadArea(CVContour quad)
    {
        double sum = 0;

        for (int i = 0; i < 4; i++)
        {
            int j = (i + 1) % 4;
            sum += quad.points[i].Item1 * quad.points[j].Item2;
            sum -= quad.points[i].Item2 * quad.points[j].Item1;
        }

        return Math.Abs(sum) * 0.5;
    }

    private static double MinDistanceSquared(CVContour quad)
    {
        double minD = distanceSquared(quad.points[0], quad.points[1]);
        minD = Math.Min(distanceSquared(quad.points[1], quad.points[2]), minD);
        minD = Math.Min(distanceSquared(quad.points[2], quad.points[3]), minD);
        minD = Math.Min(distanceSquared(quad.points[3], quad.points[0]), minD);

        return minD;
    }

    private static double QuadAspect(CVContour quad)
    {
        double a = distance(quad.points[0], quad.points[1]);
        double b = distance(quad.points[1], quad.points[2]);
        double c = distance(quad.points[2], quad.points[3]);
        double d = distance(quad.points[3], quad.points[0]);

        return (a + c) / (b + d);
    }

    static bool IsConvexQuad(CVContour quad)
    {
        int sign = 0;

        for (int i = 0; i < 4; i++)
        {
            var p0 = quad.points[i];
            var p1 = quad.points[(i + 1) % 4];
            var p2 = quad.points[(i + 2) % 4];

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

    public static void ReorderQuad(ref CVContour quad)
    {
        double dx1 = quad.points[1].Item1 - quad.points[0].Item1;
        double dy1 = quad.points[1].Item2 - quad.points[0].Item2;
        double dx2 = quad.points[2].Item1 - quad.points[0].Item1;
        double dy2 = quad.points[2].Item2 - quad.points[0].Item2;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2);

        if (crossProduct < 0.0)
        {
            (int, int) temp = quad.points[1];
            quad.points[1] = quad.points[3];
            quad.points[3] = temp;
        }

        quad.points.RemoveAt(quad.points.Count - 1);
    }

    public static List<CVContour> QuadrilateralChecks(List<CVContour> contours, int width, int height)
    {
        List<CVContour> quadrilaterals = new List<CVContour>();

        int imageArea = width * height;
        int largestDimension = Math.Max(width, height);

        double minPerimeterPixels = largestDimension * 0.03;
        double maxPerimeterPixels = largestDimension * 4.00;

        foreach (var contour in contours)
        {
            if (contour.points.Count != 5) continue;

            if (contour.perimeter < minPerimeterPixels || contour.perimeter > maxPerimeterPixels) continue;

            // double aspect = QuadAspect(contour);
            // if (aspect < 0.8 || aspect > 1.2) continue;

            if (!IsConvexQuad(contour)) continue;

            double minPixelsDistance = contour.perimeter * 0.05;
            double minCornerDistance = MinDistanceSquared(contour);
            if (minCornerDistance < minPixelsDistance * minPixelsDistance) continue;

            CVContour quad = contour;
            ReorderQuad(ref quad);

            quadrilaterals.Add(quad);
        }

        return quadrilaterals;
    }

    public static double AverageDistance(CVContour contour1, CVContour contour2)
    {
        double minDistanceSquared = double.MaxValue;
        // fc is the first corner considered on one of the markers, 4 combinations are possible
        for (int fc = 0; fc < 4; fc++)
        {
            int distanceSquared = 0;
            for (int c = 0; c < 4; c++)
            {
                // modC is the corner considering first corner is fc
                int modC = (c + fc) % 4;
                int dx = contour1.points[modC].Item1 - contour2.points[c].Item1;
                int dy = contour1.points[modC].Item2 - contour2.points[c].Item2;
                distanceSquared += dx * dx + dy * dy;
            }
            distanceSquared /= 4;
            minDistanceSquared = Math.Min(minDistanceSquared, distanceSquared);
        }
        return Math.Sqrt(minDistanceSquared);
    }

    public static double AverageModuleSize(CVContour contour, int markerSize, int markerBorderBits)
    {
        double averageArucoModuleSize = 0.0;
        for (int i = 0; i < 4; i++)
        {
            int modI = (i + 1) % 4;
            int dX = (contour.points[i].Item1 - contour.points[modI].Item1);
            int dY = (contour.points[i].Item2 - contour.points[modI].Item2);
            averageArucoModuleSize += Math.Sqrt(dX * dX + dY * dY);
        }
        int numModules = markerSize + markerBorderBits * 2;
        averageArucoModuleSize /= (4 * numModules);
        return averageArucoModuleSize;
    }

    static bool PointQuadTestOrdered((int, int) p, CVContour contour)
    {
        (int, int) p0 = contour.points[0];
        (int, int) p1 = contour.points[1];
        (int, int) p2 = contour.points[2];
        (int, int) p3 = contour.points[3];

        return SameSide(p, p0, p1, p2) &&
               SameSide(p, p1, p2, p3) &&
               SameSide(p, p2, p3, p0) &&
               SameSide(p, p3, p0, p1);
    }

    static bool SameSide((int, int) p, (int, int) a, (int, int) b, (int, int) c)
    {
        (int, int) BA = (b.Item1 - a.Item1, b.Item2 - a.Item2);
        (int, int) PA = (p.Item1 - a.Item1, p.Item2 - a.Item2);
        (int, int) CA = (c.Item1 - a.Item1, c.Item2 - a.Item2);

        float cp1 = Cross(BA, PA);
        float cp2 = Cross(BA, CA);

        return Math.Sign(cp1) == Math.Sign(cp2);
    }

    static float Cross((int, int) a, (int, int) b)
    {
        return a.Item1 * b.Item2 - a.Item2 * b.Item1;
    }

    public static bool IsInside(CVContour contour1, CVContour contour2)
    {
        return PointQuadTestOrdered(contour1.points[0], contour2) && PointQuadTestOrdered(contour1.points[1], contour2) &&
               PointQuadTestOrdered(contour1.points[2], contour2) && PointQuadTestOrdered(contour1.points[3], contour2);
    }

    public static List<CVContour> GroupContours(List<CVContour> contours, int width, int height, int markerSize)
    {
        contours.Sort((a, b) => b.perimeter.CompareTo(a.perimeter));

        List<int> groupId = Enumerable.Repeat(0, contours.Count).ToList();
        List<List<int>> groupContours = new List<List<int>>();
        List<bool> isSelectedContours = Enumerable.Repeat(true, contours.Count).ToList();

        List<List<CVContour>> closedContours = Enumerable.Repeat(new List<CVContour>(), contours.Count).ToList();

        for (int i = 0; i < contours.Count; i++)
        {
            for (int j = i + 1; j < contours.Count; j++)
            {
                double minDistance = AverageDistance(contours[i], contours[j]);
                if (minDistance < contours[j].perimeter * 0.125)
                {
                    isSelectedContours[i] = false;
                    isSelectedContours[j] = false;
                    // i and j are not related to a group
                    if (groupId[i] < 0 && groupId[j] < 0)
                    {
                        // mark candidates with their corresponding group number
                        groupId[i] = groupId[j] = (int)groupContours.Count();
                        // create group
                        groupContours.Add(new List<int>() { i, j });
                    }
                    // i is related to a group
                    else if (groupId[i] > -1 && groupId[j] == -1)
                    {
                        int group = groupId[i];
                        groupId[j] = group;
                        // add to group
                        groupContours[group].Add(j);
                    }
                    // j is related to a group
                    else if (groupId[j] > -1 && groupId[i] == -1)
                    {
                        int group = groupId[j];
                        groupId[i] = group;
                        // add to group
                        groupContours[group].Add(i);
                    }
                }
            }
            // group of one candidate
            if (isSelectedContours[i])
            {
                isSelectedContours[i] = false;
                groupId[i] = (int)groupContours.Count();
                groupContours.Add(new List<int>() { i });
            }
        }

        foreach (List<int> groupContour in groupContours)
        {
            // choose largest contours
            groupContour.Sort((a, b) => b.CompareTo(a));

            int currentId = groupContour[0];
            // check if it is too near to the image border
            bool tooNearBorder = false;
            foreach ((int, int) point in contours[currentId].points)
            {
                if (point.Item1 < 3 ||
                    point.Item2 < 3 ||
                    point.Item1 > width - 1 - 3 ||
                    point.Item2 > height - 1 - 3)
                {
                    tooNearBorder = true;
                    break;
                }
            }
            if (tooNearBorder)
            {
                continue;
            }
            isSelectedContours[currentId] = true;
            for (int i = 1; i < groupContour.Count; i++)
            {
                int id = groupContour[i];
                double dist = AverageDistance(contours[id], contours[currentId]);
                double moduleSize = AverageModuleSize(contours[id], markerSize, 1);
                if (dist > 5 * moduleSize)
                {
                    currentId = id;
                    contours[groupContour[0]].closeContours.Add(id);
                }
            }
        }

        List<CVContour> selectedCandidates = new List<CVContour>();
        for (int i = 0; i < contours.Count(); i++)
        {
            if (isSelectedContours[i])
            {
                selectedCandidates.Add(contours[i]);
            }
        }

        // find hierarchy in the candidate tree
        for (int i = selectedCandidates.Count - 1; i >= 0; i--)
        {
            for (int j = i - 1; j >= 0; j--)
            {
                if (IsInside(selectedCandidates[i], selectedCandidates[j]))
                {
                    selectedCandidates[i].parent = j;
                    selectedCandidates[j].depth = Math.Max(selectedCandidates[j].depth, selectedCandidates[i].depth + 1);
                    break;
                }
            }
        }
        return selectedCandidates;
    }

    public static CVImage ExtractPixelData(CVImage image, CVContour contour, int markerSize)
    {
        List<VectorD> srcPoints = new List<VectorD>();
        srcPoints.Add(DenseVectorD.OfArray(new double[] { contour.points[0].Item1, contour.points[0].Item2 }));
        srcPoints.Add(DenseVectorD.OfArray(new double[] { contour.points[1].Item1, contour.points[1].Item2 }));
        srcPoints.Add(DenseVectorD.OfArray(new double[] { contour.points[2].Item1, contour.points[2].Item2 }));
        srcPoints.Add(DenseVectorD.OfArray(new double[] { contour.points[3].Item1, contour.points[3].Item2 }));

        CVImage warped = CVWarp.WarpPerspective(image, srcPoints, out MatrixD _);
        warped = CVResize.ResizeImage(warped, warped.Width - 2, warped.Height - 2, CV_ResizeMode.CV_CROP_NEAREST);
        warped = CVProcessing.OtsuThreshold(warped, 256);
        warped = CVProcessing.SumWindowResample(warped, markerSize + 2, markerSize + 2);

        return warped;
    }

    /*

static Mat _extractCellPixelRatio(InputArray _image, const vector<Point2f>& corners, int markerSize,
                                   int markerBorderBits, int cellSize, double cellMarginRate, double minStdDevOtsu) {
    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(corners.size() == 4ull);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 0.5);
    CV_Assert(minStdDevOtsu >= 0);

    // number of bits in the marker
    int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
    int cellMarginPixels = int(cellMarginRate * cellSize);

    Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr<Point2f>(0)[2] =
        Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    Mat transformation = getPerspectiveTransform(corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                    INTER_NEAREST);

    // output image containing the ratio of white pixels in each cell
    Mat cellPixelRatio(markerSizeWithBorders, markerSizeWithBorders, CV_32FC1, Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all pixels are the same color (black or white)
    Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
                          .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
    meanStdDev(innerRegion, mean, stddev);
    if(stddev.ptr< double >(0)[0] < minStdDevOtsu) {
        // all black or all white, depending on mean value
        if(mean.ptr< double >(0)[0] > 127){
            cellPixelRatio.setTo(1);
        } else {
            cellPixelRatio.setTo(0);
        }

        return cellPixelRatio;
    }

    // now extract code, first threshold using Otsu
    threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // for each cell
    for(int y = 0; y < markerSizeWithBorders; y++) {
        for(int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize) + cellMarginPixels;
            int Ystart = y * (cellSize) + cellMarginPixels;
            Mat square = resultImg(Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                                        cellSize - 2 * cellMarginPixels));
            // count white pixels on each cell to assign its value
            size_t nZ = (size_t) countNonZero(square);

            // define the cell pixel ratio as the ratio of the white pixels. For inverted markers, the ratio will be inverted.
            cellPixelRatio.at<float>(y, x) = (nZ / (float)square.total());
        }
    }

    return cellPixelRatio;
}

    */

    public static bool IdentifyContour(CVImage image, CVContour contour, int markerSize)
    {
        CVImage cellPixelRatio = ExtractPixelData(image, contour, markerSize);

        return false;
    }

    /*

    CV_DbgAssert(params.markerBorderBits > 0);
        uint8_t typ=1;
        // get bits
        // scale corners to the correct size to search on the corresponding image pyramid
        vector<Point2f> scaled_corners(4);
        for (int i = 0; i < 4; ++i) {
            scaled_corners[i].x = _corners[i].x * scale;
            scaled_corners[i].y = _corners[i].y * scale;
        }

        Mat cellPixelRatio =
            _extractCellPixelRatio(_image, scaled_corners, dictionary.markerSize, params.markerBorderBits,
                                   params.perspectiveRemovePixelPerCell,
                                   params.perspectiveRemoveIgnoredMarginPerCell, params.minOtsuStdDev);

        // analyze border bits
        int maximumErrorsInBorder =
        int(dictionary.markerSize * dictionary.markerSize * params.maxErroneousBitsInBorderRate);
        int borderErrors = 0, invBorderErrors = 0;
        _getBorderErrors(cellPixelRatio, dictionary.markerSize, params.markerBorderBits, params.validBitIdThreshold,
                         borderErrors, invBorderErrors);

        // check if it is a white marker
        if(params.detectInvertedMarker && invBorderErrors < borderErrors) {
            // white marker: invert the observed ratios in place and continue as a normal marker
            borderErrors = invBorderErrors;
            subtract(Scalar::all(1), cellPixelRatio, cellPixelRatio);
            typ=2;
        }
        if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong

        // take only inner bits
        Mat onlyCellPixelRatio = cellPixelRatio(
            Rect(params.markerBorderBits, params.markerBorderBits,
                 cellPixelRatio.cols - 2 * params.markerBorderBits,
                 cellPixelRatio.rows - 2 * params.markerBorderBits));

        // try to identify the marker
        if(!dictionary.identify(onlyCellPixelRatio, idx, rotation, params.errorCorrectionRate, params.validBitIdThreshold))
            return 0;

        // compute the candidate's confidence
        if(confidenceNeeded) {
            Mat groundTruthbits = dictionary.getMarkerBits(idx, rotation);
            markerConfidence = _getMarkerConfidence(groundTruthbits, cellPixelRatio, dictionary.markerSize, params.markerBorderBits);
        }

        return typ;

    */

    public static void IdentifyContours(CVImage image, List<CVContour> contours, int markerSize)
    {
        int n = contours.Count;

        List<double> markersConfidenceTmp = Enumerable.Repeat(0.0, n).ToList();
        List<int> idsTmp = Enumerable.Repeat(-1, n).ToList();
        List<int> rotated = Enumerable.Repeat(0, n).ToList();
        List<bool> validCandidates = Enumerable.Repeat(false, n).ToList();
        List<bool> was = Enumerable.Repeat(false, n).ToList();

        int maxDepth = 0;
        for (int i = 0; i < n; i++)
            maxDepth = Math.Max(contours[i].depth, maxDepth);

        List<List<int>> depths = Enumerable.Repeat(new List<int>(), maxDepth + 1).ToList();
        for (int i = 0; i < n; i++)
            depths[contours[i].depth].Add(i);

        //// Analyze each of the candidates
        int depth = 0;
        int counter = 0;
        while (counter < n)
        {
            for (int i = 0; i < n; i++)
            {
                int v = depths[depth][i];
                was[v] = true;

                validCandidates[v] = IdentifyContour(image, contours[v], markerSize);

                if (!validCandidates[v])
                {
                    foreach (int closeContourIndex in contours[v].closeContours)
                    {
                        CVContour closeContour = contours[closeContourIndex];
                        validCandidates[v] = IdentifyContour(image, closeContour, markerSize);
                        if (validCandidates[v])
                        {
                            contours[v] = closeContour;
                            break;
                        }
                    }
                }
            }

            // visit the parent vertices of the detected markers to skip identify parent contours
            foreach (int v in depths[depth])
            {
                if (validCandidates[v])
                {
                    int parent = contours[v].parent;
                    while (parent != -1)
                    {
                        if (!was[parent])
                        {
                            was[parent] = true;
                            counter++;
                        }
                        parent = contours[parent].parent;
                    }
                }
                counter++;
            }
            depth++;
        }
    }

    /*

void identifyCandidates(const Mat& grey, const vector<Mat>& image_pyr, vector<MarkerCandidateTree>& selectedContours,
                            vector<vector<Point2f> >& accepted, vector<vector<Point> >& contours,
                            vector<int>& ids, const Dictionary& currentDictionary, vector<vector<Point2f>>& rejected, vector<float>& markersConfidence, bool confidenceNeeded) const {
        size_t ncandidates = selectedContours.size();

        vector<float> markersConfidenceTmp(ncandidates, 0.f);
        vector<int> idsTmp(ncandidates, -1);
        vector<int> rotated(ncandidates, 0);
        vector<uint8_t> validCandidates(ncandidates, 0);
        vector<uint8_t> was(ncandidates, false);
        bool checkCloseContours = true;

        int maxDepth = 0;
        for (size_t i = 0ull; i < selectedContours.size(); i++)
            maxDepth = max(selectedContours[i].depth, maxDepth);
        vector<vector<size_t>> depths(maxDepth+1);
        for (size_t i = 0ull; i < selectedContours.size(); i++) {
            depths[selectedContours[i].depth].push_back(i);
        }

        //// Analyze each of the candidates
        int depth = 0;
        size_t counter = 0;
        while (counter < ncandidates) {
            parallel_for_(Range(0, (int)depths[depth].size()), [&](const Range& range) {
                const int begin = range.start;
                const int end = range.end;
                for (int i = begin; i < end; i++) {
                    size_t v = depths[depth][i];
                    was[v] = true;
                    Mat img = grey;
                    // implements equation (4)
                    if (detectorParams.useAruco3Detection) {
                        const int minPerimeter = detectorParams.minSideLengthCanonicalImg * 4;
                        const size_t nearestImgId = _findOptPyrImageForCanonicalImg(image_pyr, grey.cols, static_cast<int>(selectedContours[v].contour.size()), minPerimeter);
                        img = image_pyr[nearestImgId];
                    }
                    const float scale = detectorParams.useAruco3Detection ? img.cols / static_cast<float>(grey.cols) : 1.f;

                    validCandidates[v] = _identifyOneCandidate(currentDictionary, img, selectedContours[v].corners, idsTmp[v], detectorParams, rotated[v], markersConfidenceTmp[v], confidenceNeeded, scale);

                    if (validCandidates[v] == 0 && checkCloseContours) {
                        for (const MarkerCandidate& closeMarkerCandidate: selectedContours[v].closeContours) {
                            validCandidates[v] = _identifyOneCandidate(currentDictionary, img, closeMarkerCandidate.corners, idsTmp[v], detectorParams, rotated[v], markersConfidenceTmp[v], confidenceNeeded, scale);
                            if (validCandidates[v] > 0) {
                                selectedContours[v].corners = closeMarkerCandidate.corners;
                                selectedContours[v].contour = closeMarkerCandidate.contour;
                                break;
                            }
                        }
                    }
                }
            });

            // visit the parent vertices of the detected markers to skip identify parent contours
            for(size_t v : depths[depth]) {
                if(validCandidates[v] > 0) {
                    int parent = selectedContours[v].parent;
                    while (parent != -1) {
                        if (!was[parent]) {
                            was[parent] = true;
                            counter++;
                        }
                        parent = selectedContours[parent].parent;
                    }
                }
                counter++;
            }
            depth++;
        }

        for (size_t i = 0ull; i < selectedContours.size(); i++) {
            if (validCandidates[i] > 0) {
                // shift corner positions to the correct rotation
                correctCornerPosition(selectedContours[i].corners, rotated[i]);

                accepted.push_back(selectedContours[i].corners);
                contours.push_back(selectedContours[i].contour);
                ids.push_back(idsTmp[i]);
            } else {
                rejected.push_back(selectedContours[i].corners);
            }
        }

        if(confidenceNeeded) {
            for (size_t i = 0ull; i < selectedContours.size(); i++) {
                if (validCandidates[i] > 0) {
                    markersConfidence.push_back(markersConfidenceTmp[i]);
                }
            }
        }
    }

    */
}