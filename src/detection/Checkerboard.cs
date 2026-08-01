using System.Runtime.ExceptionServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Microsoft.VisualBasic;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;

namespace CVNet;

public struct GridFit
{
    public bool Valid;
    public Vector<double> Center;
    public Vector<double> DirI;
    public Vector<double> DirJ;
    public Vector<double> DirII;
    public Vector<double> DirJJ;
    public Vector<double> DirIJ;
}

public class CVCheckerboard
{
    private static List<(int distance, int index)> getNearestCorners(List<(int x, int y, double score)> corners, int cornerIndex, int count)
    {
        List<(int distance, int index)> bestValues = new List<(int, int)>();
        for (int i = 0; i < count; i++)
        {
            bestValues.Add((int.MaxValue, 0));
        }

        for (int i = 0; i < corners.Count; i++)
        {
            if (i == cornerIndex) continue;

            int dx = corners[cornerIndex].x - corners[i].x;
            int dy = corners[cornerIndex].y - corners[i].y;

            int dist = dx * dx + dy * dy;
            for (int j = 0; j < count; j++)
            {
                if (dist < bestValues[j].distance)
                {
                    for (int k = count - 1; k >= j + 1; k--)
                    {
                        bestValues[k] = bestValues[k - 1];
                    }

                    bestValues[j] = (dist, i);
                    break;
                }
            }
        }

        return bestValues;
    }

    private static (double distance, int index) getNearestCorner(List<(int x, int y, double score)> corners, HashSet<int> matchedCorners, Vector<double> position)
    {
        double bestDistance = int.MaxValue;
        int bestIndex = 0;

        for (int i = 0; i < corners.Count; i++)
        {
            if (matchedCorners.Contains(i)) continue;

            double dx = position[0] - corners[i].x;
            double dy = position[1] - corners[i].y;

            double dist = dx * dx + dy * dy;
            if (dist < bestDistance)
            {
                bestDistance = dist;
                bestIndex = i;
            }
        }

        return (bestDistance, bestIndex);
    }

    private static void refineDirections(List<(int i, int j, int index)> inliers, List<(int x, int y, double score)> corners, ref GridFit fit)
    {
        int n = inliers.Count;

        if (n < 12) return;

        // Design matrix A (N x 5)
        var A = DenseMatrix.Create(n, 6, 0.0);

        for (int k = 0; k < n; k++)
        {
            int i = inliers[k].i;
            int j = inliers[k].j;

            A[k, 0] = 1;
            A[k, 1] = i;
            A[k, 2] = j;
            A[k, 3] = i * i;
            A[k, 4] = j * j;
            A[k, 5] = i * j;
        }

        // Measurement matrix Xmat (N x 2)
        var Xmat = DenseMatrix.Create(n, 2, 0.0);

        for (int k = 0; k < n; k++)
        {
            Xmat[k, 0] = corners[inliers[k].index].x;
            Xmat[k, 1] = corners[inliers[k].index].y;
        }

        // Solve A * D ≈ Xmat
        Matrix<double> D = A.QR().Solve(Xmat);

        if (D.Row(1).L2Norm() < 1) return;

        if (D.Row(2).L2Norm() < 1) return;

        fit = new GridFit()
        {
            Valid = true,
            Center = D.Row(0),
            DirI = D.Row(1),
            DirJ = D.Row(2),
            DirII = D.Row(3),
            DirJJ = D.Row(4),
            DirIJ = D.Row(5),
        };
    }

    private static List<(double x, double y, double score)> boardEstimate(List<(int x, int y, double score)> corners, GridFit fit, int offsetX, int offsetY, int patternRadius, int threshold)
    {
        List<(double x, double y, double score)> board = new List<(double x, double y, double score)>();
        HashSet<int> matchedCorners = new HashSet<int>();

        for (int i = -patternRadius + offsetX; i <= patternRadius + offsetX; i++)
        {
            Vector<double> searchI = fit.Center + i * fit.DirI + i * i * fit.DirII;

            for (int j = -patternRadius + offsetY; j <= patternRadius + offsetY; j++)
            {
                Vector<double> search = searchI + j * fit.DirJ + j * j * fit.DirJJ + i * j * fit.DirIJ;

                (double distance, int index) corner = getNearestCorner(corners, matchedCorners, search);

                if (corner.distance < threshold)
                {
                    board.Add((corners[corner.index].x, corners[corner.index].y, 1.0));
                    matchedCorners.Add(corner.index);
                }
                else
                {
                    board.Add((search[0], search[1], 0.3));
                }
            }
        }

        return board;
    }

    private static double checkInliers(List<(int x, int y, double score)> corners, GridFit fit, int patternRadius, int threshold, out List<(int i, int j, int index)> inliers, out (int, int, int, int) inlierBounds)
    {
        inliers = new List<(int i, int j, int index)>();
        double allError = 0;
        HashSet<int> matchedCorners = new HashSet<int>();

        int minInlierX = 0;
        int maxInlierX = 0;
        int minInlierY = 0;
        int maxInlierY = 0;

        for (int i = -patternRadius; i <= patternRadius; i++)
        {
            Vector<double> searchI = fit.Center + i * fit.DirI + i * i * fit.DirII;

            for (int j = -patternRadius; j <= patternRadius; j++)
            {
                Vector<double> search = searchI + j * fit.DirJ + j * j * fit.DirJJ + i * j * fit.DirIJ;

                (double distance, int index) corner = getNearestCorner(corners, matchedCorners, search);

                if (corner.distance < threshold)
                {
                    if (i < minInlierX) minInlierX = i;
                    if (i > maxInlierX) maxInlierX = i;
                    if (j < minInlierY) minInlierY = j;
                    if (j > maxInlierY) maxInlierY = j;

                    allError += corner.distance;
                    inliers.Add((i, j, corner.index));
                    matchedCorners.Add(corner.index);
                }
            }
        }

        inlierBounds = (minInlierX, minInlierY, maxInlierX, maxInlierY);

        return allError;
    }

    private static double checkInliersRefined(List<(int x, int y, double score)> corners, ref GridFit fit, int patternRadius, int threshold, out List<(int i, int j, int index)> inliers, out (int, int, int, int) inlierBounds)
    {
        // Gradually expand pattern size while refining directions
        // This prevents selecting diagonals while keeping fine directions
        for (int i = 2; i < patternRadius; i++)
        {
            // Estimate inliers with current pattern size and direction
            checkInliers(corners, fit, i, threshold, out List<(int i, int j, int index)> tempInliers, out _);

            // Refine direction with current pattern
            refineDirections(tempInliers, corners, ref fit);
        }

        return checkInliers(corners, fit, patternRadius, threshold, out inliers, out inlierBounds);
    }

    private static GridFit initialGuess(List<(int x, int y, double score)> corners, int cornerIndex)
    {
        int centerX = corners[cornerIndex].x;
        int centerY = corners[cornerIndex].y;

        List<(int distance, int index)> nearestCorners = getNearestCorners(corners, cornerIndex, 4);

        (double x, double y) dir1 = (corners[nearestCorners[0].index].x - centerX, corners[nearestCorners[0].index].y - centerY);
        double l1 = Math.Sqrt(dir1.x * dir1.x + dir1.y * dir1.y);

        double minAngle = 1.0;
        (double x, double y) dir2 = (0, 0);

        for (int j = 1; j < nearestCorners.Count; j++)
        {
            (double x, double y) dirN = (corners[nearestCorners[j].index].x - centerX, corners[nearestCorners[j].index].y - centerY);
            double lN = Math.Sqrt(dirN.x * dirN.x + dirN.y * dirN.y);

            double dot1N = dir1.x * dirN.x + dir1.y * dirN.y;

            double angle = Math.Abs(dot1N / (l1 * lN));

            if (angle < minAngle && angle > 0.1 && angle < 0.9)
            {
                minAngle = angle;
                dir2 = dirN;
            }
        }

        if (minAngle == 1.0)
        {
            return new GridFit()
            {
                Valid = false,
            };
        }

        return new GridFit()
        {
            Valid = true,
            Center = DenseVector.OfArray([centerX, centerY]),
            DirI = DenseVector.OfArray([dir1.x, dir1.y]),
            DirJ = DenseVector.OfArray([dir2.x, dir2.y]),
            DirII = DenseVector.OfArray([0, 0]),
            DirJJ = DenseVector.OfArray([0, 0]),
            DirIJ = DenseVector.OfArray([0, 0]),
        };
    }

    public static List<Vector<double>> DetectCheckerboard(CVImage image, int radius, double threshold, int nonMaxSuppressionRadius, int maxPixelOffset, int patternRadius)
    {
        List<Vector<double>> saddlePoints = new List<Vector<double>>();

        List<(int x, int y, double score)> corners = CVCornerDetector.DetectCornerShiTomasi(image, radius, threshold, nonMaxSuppressionRadius);

        List<(int i, int j, int index)> maxInliers = new List<(int i, int j, int index)>();
        (int, int, int, int) bestInlierBounds = (0, 0, 0, 0);
        double bestError = double.MaxValue;
        GridFit bestFit = new();

        int maxPixelOffsetSquared = maxPixelOffset * maxPixelOffset;

        for (int i = 0; i < corners.Count; i++)
        {
            GridFit fit = initialGuess(corners, i);

            if (!fit.Valid) continue;

            double error = checkInliersRefined(corners, ref fit, patternRadius, maxPixelOffsetSquared, out List<(int i, int j, int index)> inliers, out var inlierBounds);

            if (inliers.Count > maxInliers.Count || (inliers.Count == maxInliers.Count && error < bestError))
            {
                maxInliers = inliers;
                bestInlierBounds = inlierBounds;
                bestFit = fit;
                bestError = error;
            }
        }

        if (maxInliers.Count < (patternRadius * 2 - 1) * (patternRadius * 2 - 1) * 0.7) return new();

        int offsetX = 0;
        int offsetY = 0;

        if (bestInlierBounds.Item1 == -4 && bestInlierBounds.Item3 == 3)
            offsetX -= 1;
        if (bestInlierBounds.Item1 == -3 && bestInlierBounds.Item3 == 4)
            offsetX += 1;
        if (bestInlierBounds.Item2 == -4 && bestInlierBounds.Item4 == 3)
            offsetY -= 1;
        if (bestInlierBounds.Item2 == -3 && bestInlierBounds.Item4 == 4)
            offsetY += 1;

        if (bestInlierBounds.Item1 == -4 && bestInlierBounds.Item3 == 2)
            offsetX -= 2;
        if (bestInlierBounds.Item1 == -2 && bestInlierBounds.Item3 == 4)
            offsetX += 2;
        if (bestInlierBounds.Item2 == -4 && bestInlierBounds.Item4 == 2)
            offsetY -= 2;
        if (bestInlierBounds.Item2 == -2 && bestInlierBounds.Item4 == 4)
            offsetY += 2;

        List<(double x, double y, double score)> board = boardEstimate(corners, bestFit, offsetX, offsetY, patternRadius, maxPixelOffsetSquared);

        for (int i = 0; i < board.Count; i++) saddlePoints.Add(DenseVector.OfArray([board[i].x, board[i].y, board[i].score]));
        //for (int i = 0; i < corners.Count; i++) saddlePoints.Add(DenseVector.OfArray([corners[i].Item1, corners[i].Item2, 0.2]));

        Console.WriteLine($"Saddle points: {maxInliers.Count}");
        Console.WriteLine($"Center: {bestFit.Center[0]} {bestFit.Center[1]}");
        Console.WriteLine($"DirI: {bestFit.DirI[0]} {bestFit.DirI[1]}");
        Console.WriteLine($"DirJ: {bestFit.DirJ[0]} {bestFit.DirJ[1]}");
        Console.WriteLine($"DirII: {bestFit.DirII[0]} {bestFit.DirII[1]}");
        Console.WriteLine($"DirJJ: {bestFit.DirJJ[0]} {bestFit.DirJJ[1]}");
        Console.WriteLine($"DirIJ: {bestFit.DirIJ[0]} {bestFit.DirIJ[1]}");
        Console.WriteLine($"Inlier Bounds: {bestInlierBounds.Item1} {bestInlierBounds.Item2} {bestInlierBounds.Item3} {bestInlierBounds.Item4}");
        return saddlePoints;
    }
}