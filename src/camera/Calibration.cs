namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;

public class CVCalibration
{
    public static double Calibrate(List<List<VectorD>> imagePointGroups, List<List<VectorD>> worldPointGroups, out MatrixD K, out VectorD d)
    {
        if (imagePointGroups.Count == 0 || imagePointGroups.Count != worldPointGroups.Count) throw new Exception("Invalid number of samples");

        List<MatrixD> homographies = new List<MatrixD>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            if (imagePointGroups[i].Count == 0 || imagePointGroups[i].Count != worldPointGroups[i].Count) throw new Exception("Invalid number of points");

            homographies.Add(CVCamera.ComputeHomography(imagePointGroups[i], worldPointGroups[i]));
        }

        K = CVCamera.ComputeIntrinsics(homographies);

        CVCamera.ComputeExtrinsics(homographies, K, out List<MatrixD> Rs, out List<VectorD> ts);

        d = DenseVectorD.OfArray([0, 0, 0, 0, 0]);

        List<List<VectorD>> transformedWorldPoints = CVCamera.TransformPoints(worldPointGroups, Rs, ts);

        List<VectorD> imagePoints = new List<VectorD>();
        List<VectorD> worldPoints = new List<VectorD>();

        for (int i = 0; i < imagePointGroups.Count; i++)
        {
            foreach (VectorD point in imagePointGroups[i])
                imagePoints.Add(point);
            foreach (VectorD point in transformedWorldPoints[i])
                worldPoints.Add(point);
        }

        List<VectorD> projectedWorldPoints = CVProjection.ProjectPoints(worldPoints, K, d);

        Console.WriteLine(CVCamera.ComputeError(imagePoints, projectedWorldPoints));

        CVCamera.LevenbergMarquardtWithK(worldPoints, imagePoints, ref K, ref d);

        return CVCamera.ComputeError(imagePoints, projectedWorldPoints);
    }
};