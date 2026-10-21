namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public interface CVITransform
{
    public CVITransform Inverse();

    public CVImage Apply(CVImage image);
    public VectorD Apply(VectorD vector);
};

public class CVRectTransform
{
    public int SourceWidth;
    public int SourceHeight;

    public int TargetWidth;
    public int TargetHeight;

    public CVRectTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        SourceWidth = sourceWidth;
        SourceHeight = sourceHeight;
        TargetWidth = targetWidth;
        TargetHeight = targetHeight;
    }

    public CVRectTransform Inverse()
    {
        return new CVRectTransform(
            TargetWidth,
            TargetHeight,
            SourceWidth,
            SourceHeight);
    }
};

public class CVStretchTransform : CVITransform
{
    public CVRectTransform RectTransform;
    public CVInterpolationMode Mode;

    public CVStretchTransform(CVRectTransform rectTransform, CVInterpolationMode mode)
    {
        RectTransform = rectTransform;
        Mode = mode;
    }

    public CVStretchTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, CVInterpolationMode mode)
    {
        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
        Mode = mode;
    }

    public CVImage Apply(CVImage image)
    {
        return CVResize.Stretch(image, RectTransform.TargetWidth, RectTransform.TargetHeight, Mode);
    }

    public VectorD Apply(VectorD vector)
    {
        return CVResizeTransformation.Stretch(RectTransform.SourceWidth, RectTransform.SourceHeight, vector, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public CVITransform Inverse()
    {
        return new CVStretchTransform(RectTransform.Inverse(), Mode);
    }
}

public class CVCenterCopyTransform : CVITransform
{
    public CVRectTransform RectTransform;

    public CVCenterCopyTransform(CVRectTransform rectTransform)
    {
        RectTransform = rectTransform;
    }

    public CVCenterCopyTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
    }

    public CVImage Apply(CVImage image)
    {
        return CVResize.CenterCopy(image, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public VectorD Apply(VectorD vector)
    {
        return CVResizeTransformation.CopyCenter(RectTransform.SourceWidth, RectTransform.SourceHeight, vector, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public CVITransform Inverse()
    {
        return new CVCenterCopyTransform(RectTransform.Inverse());
    }
}

public class CVAspectCropTransform : CVITransform
{
    public CVRectTransform RectTransform;

    public CVAspectCropTransform(CVRectTransform rectTransform)
    {
        RectTransform = rectTransform;
    }

    public CVAspectCropTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
    }

    public CVAspectCropTransform(int sourceWidth, int sourceHeight, double targetAspect)
    {
        CVResize.GetAspectCropDimensions(
            sourceWidth,
            sourceHeight,
            targetAspect,
            out int targetWidth,
            out int targetHeight);

        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
    }

    public CVImage Apply(CVImage image)
    {
        return CVResize.CenterCopy(image, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public VectorD Apply(VectorD vector)
    {
        return CVResizeTransformation.CopyCenter(RectTransform.SourceWidth, RectTransform.SourceHeight, vector, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public CVITransform Inverse()
    {
        return new CVCenterCopyTransform(RectTransform.Inverse());
    }
}

public class CVAspectPadTransform : CVITransform
{
    public CVRectTransform RectTransform;

    public CVAspectPadTransform(CVRectTransform rectTransform)
    {
        RectTransform = rectTransform;
    }

    public CVAspectPadTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
    {
        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
    }

    public CVAspectPadTransform(int sourceWidth, int sourceHeight, double targetAspect)
    {
        CVResize.GetAspectPadDimensions(
            sourceWidth,
            sourceHeight,
            targetAspect,
            out int targetWidth,
            out int targetHeight);

        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
    }

    public CVImage Apply(CVImage image)
    {
        return CVResize.CenterCopy(image, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public VectorD Apply(VectorD vector)
    {
        return CVResizeTransformation.CopyCenter(RectTransform.SourceWidth, RectTransform.SourceHeight, vector, RectTransform.TargetWidth, RectTransform.TargetHeight);
    }

    public CVITransform Inverse()
    {
        return new CVCenterCopyTransform(RectTransform.Inverse());
    }
}

public class CVWarpTransform : CVITransform
{
    public CVRectTransform RectTransform;
    public MatrixD TransformationMatrix;

    public CVWarpTransform(CVRectTransform rectTransform, MatrixD transformationMatrix)
    {
        RectTransform = rectTransform;
        TransformationMatrix = transformationMatrix;
    }

    public CVWarpTransform(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight, MatrixD transformationMatrix)
    {
        RectTransform = new CVRectTransform(
            sourceWidth,
            sourceHeight,
            targetWidth,
            targetHeight);
        TransformationMatrix = transformationMatrix;
    }

    public CVImage Apply(CVImage image)
    {
        return CVWarp.WarpPerspective(image, RectTransform.TargetWidth, RectTransform.TargetHeight, TransformationMatrix);
    }

    public VectorD Apply(VectorD vector)
    {
        return CVProjection.HomographyProjectPoint(vector, TransformationMatrix);
    }

    public CVITransform Inverse()
    {
        return new CVWarpTransform(RectTransform.Inverse(), TransformationMatrix.Inverse());
    }
}

public class CVTransformList : CVITransform
{
    public List<CVITransform> Transforms = new List<CVITransform>();

    public CVImage Apply(CVImage image)
    {
        CVImage tempImage = image;
        foreach (CVITransform transform in Transforms)
            tempImage = transform.Apply(tempImage);
        return tempImage;
    }

    public VectorD Apply(VectorD vector)
    {
        VectorD tempVector = vector;
        foreach (CVITransform transform in Transforms)
            tempVector = transform.Apply(tempVector);
        return tempVector;
    }

    public CVITransform Inverse()
    {
        List<CVITransform> inverseList = new List<CVITransform>();

        for (int i = Transforms.Count - 1; i >= 0; i--)
            inverseList.Add(Transforms[i].Inverse());

        return new CVTransformList()
        {
            Transforms = inverseList,
        };
    }
}