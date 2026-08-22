namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public abstract class CVTransform
{
    public abstract CVTransform Inverse();
};

public abstract class CVRectTransform : CVTransform
{
    public int SourceWidth;
    public int SourceHeight;

    public int TargetWidth;
    public int TargetHeight;
};

public class CVStretchTransform : CVRectTransform
{
    public override CVTransform Inverse()
    {
        return new CVStretchTransform()
        {
            SourceWidth = TargetWidth,
            SourceHeight = TargetHeight,
            TargetWidth = SourceWidth,
            TargetHeight = SourceHeight,
        };
    }
}

public class CVCropTransform : CVRectTransform
{
    public override CVTransform Inverse()
    {
        return new CVPadTransform()
        {
            SourceWidth = TargetWidth,
            SourceHeight = TargetHeight,
            TargetWidth = SourceWidth,
            TargetHeight = SourceHeight,
        };
    }
}

public class CVPadTransform : CVRectTransform
{
    public override CVTransform Inverse()
    {
        return new CVCropTransform()
        {
            SourceWidth = TargetWidth,
            SourceHeight = TargetHeight,
            TargetWidth = SourceWidth,
            TargetHeight = SourceHeight,
        };
    }
}

public class CVWarpTransform : CVTransform
{
    public MatrixD Warp = DenseMatrixD.CreateIdentity(3);

    public override CVTransform Inverse()
    {
        return new CVWarpTransform()
        {
            Warp = Warp.Inverse(),
        };
    }
};