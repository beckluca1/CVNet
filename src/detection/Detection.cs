using System.Drawing;
using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public class CVDetection
{
    // Based on DouglasPeucker Algortihm
    private static List<(int, int)> ApproximatePolygons(
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
            var left = ApproximatePolygons(points.GetRange(first, index - first + 1), epsilon);
            var right = ApproximatePolygons(points.GetRange(index, last - index + 1), epsilon);

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

    public static double distanceSquared((int x, int y) a, (int x, int y) b)
    {
        int dX = b.x - a.x;
        int dY = b.y - a.y;
        return dX * dX + dY * dY;
    }

    public static List<CVContour> ApproximatePolygons(List<CVContour> contours, double epsilon)
    {
        List<CVContour> polygons = new List<CVContour>();

        foreach (var contour in contours)
        {
            List<(int x, int y)> approx = ApproximatePolygons(contour.points, epsilon * contour.points.Count);

            //Remove last element when at least a triangle
            if (approx.Count > 3)
                approx.RemoveAt(approx.Count - 1);

            polygons.Add(new CVContour(approx, contour.points.Count));
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
    }

    public static List<CVContour> ContourChecks(List<CVContour> contours, int width, int height)
    {
        List<CVContour> checkedContours = new List<CVContour>();

        int largestDimension = Math.Max(width, height);

        double minPerimeterPixels = largestDimension * 0.03;
        double maxPerimeterPixels = largestDimension * 4.00;

        foreach (var contour in contours)
        {
            if (contour.points.Count < minPerimeterPixels || contour.points.Count > maxPerimeterPixels) continue;

            checkedContours.Add(contour);
        }

        return checkedContours;
    }

    public static List<CVContour> QuadrilateralChecks(List<CVContour> contours, int width, int height)
    {
        double minAre = width * height * 0.0001;

        List<CVContour> quadrilaterals = new List<CVContour>();

        foreach (var contour in contours)
        {
            if (contour.points.Count != 4) continue;

            // double aspect = QuadAspect(contour);

            //double area = QuadArea(contour);

            //if (area < minAre) continue;

            // if (aspect < 0.8 || aspect > 1.2) continue;

            if (!IsConvexQuad(contour)) continue;

            double minPixelsDistance = contour.PixelCount * 0.05;
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
            int dX = contour.points[i].Item1 - contour.points[modI].Item1;
            int dY = contour.points[i].Item2 - contour.points[modI].Item2;
            averageArucoModuleSize += Math.Sqrt(dX * dX + dY * dY);
        }
        int numModules = markerSize + markerBorderBits * 2;
        averageArucoModuleSize /= 4 * numModules;
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

    public static List<int> GroupContours(List<CVContour> contours, int width, int height, int markerSize)
    {
        contours.Sort((a, b) => b.PixelCount.CompareTo(a.PixelCount));

        List<int> groupIds = Enumerable.Repeat(-1, contours.Count).ToList();
        List<List<int>> contourGroups = new List<List<int>>();
        List<bool> isSelectedContours = Enumerable.Repeat(true, contours.Count).ToList();

        List<List<CVContour>> closedContours = Enumerable.Repeat(new List<CVContour>(), contours.Count).ToList();

        for (int i = 0; i < contours.Count; i++)
        {
            for (int j = i + 1; j < contours.Count; j++)
            {
                double minDistance = AverageDistance(contours[i], contours[j]);
                if (minDistance < contours[j].PixelCount * 0.125)
                {
                    isSelectedContours[i] = false;
                    isSelectedContours[j] = false;
                    // i and j are not related to a group
                    if (groupIds[i] == -1 && groupIds[j] == -1)
                    {
                        int newGroupId = contourGroups.Count();
                        // mark candidates with their corresponding group number
                        groupIds[i] = newGroupId;
                        groupIds[j] = newGroupId;
                        // create group
                        contourGroups.Add(new List<int>() { i, j });
                    }
                    // i is related to a group
                    else if (groupIds[i] > -1 && groupIds[j] == -1)
                    {
                        int groupId = groupIds[i];
                        groupIds[j] = groupId;
                        // add to group
                        contourGroups[groupId].Add(j);
                    }
                    // j is related to a group
                    else if (groupIds[j] > -1 && groupIds[i] == -1)
                    {
                        int groupId = groupIds[j];
                        groupIds[i] = groupId;
                        // add to group
                        contourGroups[groupId].Add(i);
                    }
                }
            }

            // No close element is found
            // group candidate
            if (isSelectedContours[i])
            {
                isSelectedContours[i] = false;
                int newGroupId = contourGroups.Count();
                groupIds[i] = newGroupId;
                contourGroups.Add(new List<int>() { i });
            }
        }

        foreach (List<int> groupContour in contourGroups)
        {
            // choose largest contours
            groupContour.Sort((a, b) => contours[b].PixelCount.CompareTo(contours[a].PixelCount));

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

        // Only add biggest contour of group
        List<int> selectedCandidates = new List<int>();
        for (int i = 0; i < contours.Count(); i++)
        {
            if (isSelectedContours[i])
            {
                selectedCandidates.Add(i);
            }
        }

        // find hierarchy in the candidate tree
        for (int i = selectedCandidates.Count - 1; i >= 0; i--)
        {
            for (int j = i - 1; j >= 0; j--)
            {
                if (IsInside(contours[selectedCandidates[i]], contours[selectedCandidates[j]]))
                {
                    contours[selectedCandidates[i]].parent = j;
                    contours[selectedCandidates[j]].depth = Math.Max(contours[selectedCandidates[j]].depth, contours[selectedCandidates[i]].depth + 1);
                    break;
                }
            }
        }
        return selectedCandidates;
    }

    public static CVImage ExtractPixelData(CVImage image, CVContour contour, int markerSize)
    {
        List<VectorD> srcPoints =
        [
            DenseVectorD.OfArray([contour.points[0].Item1, contour.points[0].Item2]),
            DenseVectorD.OfArray([contour.points[1].Item1, contour.points[1].Item2]),
            DenseVectorD.OfArray([contour.points[2].Item1, contour.points[2].Item2]),
            DenseVectorD.OfArray([contour.points[3].Item1, contour.points[3].Item2]),
        ];


        CVImage warped = CVWarp.WarpPerspective(image, srcPoints, out MatrixD _);
        warped = CVProcessing.SumWindowResample(warped, markerSize + 2, markerSize + 2);
        warped = CVThreshold.OtsuThreshold(warped, 256);
        warped = CVConvert.ConvertDataFormat(warped, image.DataFormat);

        return warped;
    }

    public static List<CVContour> IdentifyContours(CVImage image, List<CVContour> contours, List<int> contourGroups, int markerSize)
    {
        int n = contours.Count;

        List<bool> validCandidates = Enumerable.Repeat(false, n).ToList();
        List<bool> was = Enumerable.Repeat(false, n).ToList();

        List<CVContour> validContours = new List<CVContour>();

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

                validCandidates[v] = CVAruco.IdentifyContour(image, contours[v], markerSize);

                if (validCandidates[v]) validContours.Add(contours[v]);
                else
                {
                    foreach (int closeContourIndex in contours[v].closeContours)
                    {
                        CVContour closeContour = contours[closeContourIndex];
                        validCandidates[v] = CVAruco.IdentifyContour(image, closeContour, markerSize);
                        if (validCandidates[v])
                        {
                            validContours.Add(closeContour);
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

        return validContours;
    }
}