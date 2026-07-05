namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using System.Numerics;

public class CVCornerDetector
{
    private static void DerivativeX<T>(
        CVImage image,
        ref CVImage outImage) where T : struct, INumber<T>
    {
        T Two = T.One + T.One;

        T[,] k =
        {
            { -T.One, T.Zero, T.One },
            { -Two, T.Zero, Two },
            { -T.One, T.Zero, T.One }
        };

        Span<T> srcSpan = image.BufferAs<T>();
        Span<T> dstSpan = outImage.BufferAs<T>();

        for (int y = 1; y < image.Height - 1; y++)
        {
            for (int x = 1; x < image.Width - 1; x++)
            {
                T sum = T.Zero;

                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        sum +=
                            srcSpan[(x + i) + (y + j) * image.Width] *
                            k[j + 1, i + 1];
                    }
                }
                dstSpan[x + y * outImage.Width] = sum;
            }
        }
    }

    public static CVImage DerivativeX(CVImage image)
    {
        // Derivative requires signed image and could be twice as large
        CVImage imageC = CVConvert.ConvertDataFormatToSigned(image);
        imageC = CVConvert.ConvertDataFormatFactor(image, 2);
        CVImage outImage = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (imageC.DataFormat == CVDataFormat.CV_U8) DerivativeX<byte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S8) DerivativeX<sbyte>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U16) DerivativeX<ushort>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S16) DerivativeX<short>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U32) DerivativeX<uint>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S32) DerivativeX<int>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_U64) DerivativeX<ulong>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_S64) DerivativeX<long>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F32) DerivativeX<float>(imageC, ref outImage);
        else if (imageC.DataFormat == CVDataFormat.CV_F64) DerivativeX<double>(imageC, ref outImage);

        return outImage;
    }

    private static void DerivativeY<T>(
        CVImage image,
        ref CVImage outImage) where T : struct, INumber<T>
    {
        T Two = T.One + T.One;

        T[,] k =
        {
        { -T.One, -Two, -T.One },
        {  T.Zero,  T.Zero,  T.Zero },
        {  T.One,  Two,  T.One }
    };

        Span<T> srcSpan = image.BufferAs<T>();
        Span<T> dstSpan = outImage.BufferAs<T>();

        for (int y = 1; y < image.Height - 1; y++)
        {
            for (int x = 1; x < image.Width - 1; x++)
            {
                T sum = T.Zero;

                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        sum +=
                            srcSpan[(x + i) + (y + j) * image.Width] *
                            k[j + 1, i + 1];
                    }
                }
                dstSpan[x + y * outImage.Width] = sum;
            }
        }
    }

    public static CVImage DerivativeY(CVImage image)
    {
        // Derivative requires signed image
        CVImage imageC = CVConvert.ConvertDataFormatToSigned(image);
        imageC = CVConvert.ConvertDataFormatFactor(image, 2);
        CVImage outImage = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) DerivativeY<byte>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) DerivativeY<sbyte>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) DerivativeY<ushort>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) DerivativeY<short>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) DerivativeY<uint>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) DerivativeY<int>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) DerivativeY<ulong>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) DerivativeY<long>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) DerivativeY<float>(imageC, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) DerivativeY<double>(imageC, ref outImage);

        return outImage;
    }

    public static void Sobel<T>(
        CVImage image,
        ref CVImage gradientX,
        ref CVImage gradientY) where T : struct, INumber<T>
    {
        T Two = T.One + T.One;

        T[,] sx =
        {
            { -T.One, T.Zero, T.One },
            { -Two, T.Zero, Two },
            { -T.One, T.Zero, T.One }
        };

        T[,] sy =
        {
            { -T.One, -Two, -T.One },
            {  T.Zero,  T.Zero,  T.Zero },
            {  T.One,  Two,  T.One }
        };

        Span<T> bufferSpan = image.BufferAs<T>();
        Span<T> bgx = gradientX.BufferAs<T>();
        Span<T> bgy = gradientY.BufferAs<T>();

        for (int y = 1; y < image.Height - 1; y++)
            for (int x = 1; x < image.Width - 1; x++)
            {
                T sumX = T.Zero;
                T sumY = T.Zero;

                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++)
                    {
                        T p = bufferSpan[(x + i) + (y + j) * image.Width];

                        sumX += p * sx[j + 1, i + 1];
                        sumY += p * sy[j + 1, i + 1];
                    }
                bgx[x + y * gradientX.Width] = sumX;
                bgy[x + y * gradientY.Width] = sumY;
            }
    }

    public static void Sobel(CVImage image, out CVImage gradientX, out CVImage gradientY)
    {
        // Sobel requires signed image
        CVImage imageC = CVConvert.ConvertDataFormatToSigned(image);
        imageC = CVConvert.ConvertDataFormatFactor(image, 2);
        gradientX = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);
        gradientY = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) Sobel<byte>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_S8) Sobel<sbyte>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_U16) Sobel<ushort>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_S16) Sobel<short>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_U32) Sobel<uint>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_S32) Sobel<int>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_U64) Sobel<ulong>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_S64) Sobel<long>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_F32) Sobel<float>(imageC, ref gradientX, ref gradientY);
        else if (image.DataFormat == CVDataFormat.CV_F64) Sobel<double>(imageC, ref gradientX, ref gradientY);
    }

    public static void HessianEigenvalues<T>(
        CVImage image,
        CVImage gradientX,
        CVImage gradientY,
        int windowRadius,
        ref CVImage determinant,
        ref CVImage structureDeterminant,
        ref CVImage lambda1,
        ref CVImage lambda2) where T : struct, INumber<T>
    {
        CVImage ixx = DerivativeX(gradientX);
        CVImage iyy = DerivativeY(gradientY);
        CVImage ixy = DerivativeY(gradientX);

        determinant = (ixx * iyy) - (ixy * ixy);

        // Box filters
        CVImage gxxImage = CVProcessing.SumWindow(gradientX, windowRadius);
        CVImage gyyImage = CVProcessing.SumWindow(gradientY, windowRadius);

        CVImage structureAImage = gxxImage * gxxImage;
        CVImage structureBImage = gxxImage * gyyImage;
        CVImage structureCImage = gyyImage * gyyImage;

        CVImage structureDetImageC = ((structureAImage * structureCImage) - (structureBImage * structureBImage)) * 4;

        CVImage structureTraceImage = structureAImage + structureCImage;
        CVImage structureTraceImageSquared = structureTraceImage * structureTraceImage;

        CVImage discriminantSquared = structureTraceImageSquared - structureDetImageC;
        discriminantSquared = CVMax.Max(discriminantSquared, 0);
        CVImage discriminant = CVSquareRoot.SquareRoot(discriminantSquared);

        lambda1 = (structureTraceImage + discriminant) / 2;
        lambda2 = (structureTraceImage - discriminant) / 2;
    }

    public static void HessianEigenvalues(CVImage image, out CVImage determinant, out CVImage structureDeterminant, out CVImage lambda1, out CVImage lambda2, int constant = 25, int windowRadius = 3)
    {
        // Hessian requires signed image
        CVImage imageC = CVConvert.ConvertDataFormatToSigned(image);
        Sobel(imageC, out CVImage gradientX, out CVImage gradientY);

        determinant = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);
        structureDeterminant = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);
        lambda1 = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);
        lambda2 = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) HessianEigenvalues<byte>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_S8) HessianEigenvalues<sbyte>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_U16) HessianEigenvalues<ushort>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_S16) HessianEigenvalues<short>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_U32) HessianEigenvalues<uint>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_S32) HessianEigenvalues<int>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_U64) HessianEigenvalues<ulong>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_S64) HessianEigenvalues<long>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_F32) HessianEigenvalues<float>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
        else if (image.DataFormat == CVDataFormat.CV_F64) HessianEigenvalues<double>(image, gradientX, gradientY, windowRadius, ref determinant, ref structureDeterminant, ref lambda1, ref lambda2);
    }

    public static List<(int x, int y, double score)> NonMaximumSuppression(
        List<(int x, int y, double score)> points,
        float radius = 8)
    {
        List<(int x, int y, double score)> result = new List<(int x, int y, double score)>();

        foreach (var p in points.OrderByDescending(v => v.score))
        {
            bool keep = true;

            foreach (var q in result)
            {
                double dx = p.x - q.x;
                double dy = p.y - q.x;

                if (dx * dx + dy * dy <
                    radius * radius)
                {
                    keep = false;
                    break;
                }
            }

            if (keep)
                result.Add(p);
        }

        return result;
    }

    public static List<(int, int, double)> DetectCornerHessian(CVImage image, int constant1 = 2, int constant2 = 5, int threshold = 1)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        gray = CVConvert.ConvertDataFormat(image, CVDataFormat.CV_F32);
        gray = CVDivide.Divide(gray, 255f);
        gray = CVBlur.GaussianBlur(gray, 5);

        HessianEigenvalues(gray, out CVImage determinant, out CVImage structureDeterminant, out CVImage lambda1, out CVImage lambda2);
        CVImage determinantMask = CVBigger.Bigger(determinant, -threshold);
        CVImage structureDeterminantMask = CVBigger.Bigger(structureDeterminant, 0);
        CVImage lambdaMask = CVBigger.Bigger(lambda1 * constant1, lambda2 * constant2);
        CVImage hessianMask = determinantMask * structureDeterminantMask * lambdaMask;
        CVImage hessianStrength = determinant * (-1);
        var pixelList = CVProcessing.GetPixels(hessianMask, hessianStrength, 1);
        return NonMaximumSuppression(pixelList);
    }

    private static void harrisStrength<T>(
            CVImage gradientX,
            CVImage gradientY,
            T constant,
            int windowRadius,
            ref CVImage outImage) where T : struct, INumber<T>
    {
        // Build products Ix^2, Iy^2, IxIy
        CVImage IxxImage = gradientX * gradientX;
        CVImage IyyImage = gradientY * gradientY;
        CVImage IxyImage = gradientX * gradientY;

        // Box Filters
        CVImage SxxImage = CVProcessing.SumWindow(IxxImage, windowRadius);
        CVImage SyyImage = CVProcessing.SumWindow(IyyImage, windowRadius);
        CVImage SxyImage = CVProcessing.SumWindow(IxyImage, windowRadius);

        // Harris Formula
        CVImage detImageC = ((SxxImage * SyyImage) - (SxyImage * SxyImage)) * 25;
        CVImage traceImage = SxxImage + SyyImage;
        CVImage traceImageSquared = traceImage * traceImage;

        outImage = detImageC - traceImageSquared;
    }

    public static CVImage HarrisStrength(CVImage image, int constant = 25, int windowRadius = 3)
    {
        // Harris requires signed image
        CVImage imageC = CVConvert.ConvertDataFormatToSigned(image);
        Sobel(imageC, out CVImage gradientX, out CVImage gradientY);
        CVImage outImage = CVImage.Create(imageC.Width, imageC.Height, imageC.ColorFormat, imageC.DataFormat, imageC.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) harrisStrength<byte>(gradientX, gradientY, (byte)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) harrisStrength<sbyte>(gradientX, gradientY, (sbyte)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) harrisStrength<ushort>(gradientX, gradientY, (ushort)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) harrisStrength<short>(gradientX, gradientY, (short)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) harrisStrength<uint>(gradientX, gradientY, (uint)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) harrisStrength<int>(gradientX, gradientY, (int)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) harrisStrength<ulong>(gradientX, gradientY, (ulong)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) harrisStrength<long>(gradientX, gradientY, (long)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) harrisStrength<float>(gradientX, gradientY, (float)constant, windowRadius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) harrisStrength<double>(gradientX, gradientY, (double)constant, windowRadius, ref outImage);

        return outImage;
    }

    public static List<(int, int, double)> DetectCornersHarris(
        CVImage image,
        int constant = 25,
        int windowRadius = 3,
        int threshold = 1)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);
        gray = CVConvert.ConvertDataFormat(image, CVDataFormat.CV_F32);
        gray = CVDivide.Divide(gray, 255f);
        gray = CVBlur.GaussianBlur(gray, 5);

        CVImage harrisStrength = HarrisStrength(gray, constant, windowRadius);
        CVImage harrisMask = CVBigger.Bigger(harrisStrength, threshold);
        var pixelList = CVProcessing.GetPixels(harrisMask, harrisStrength, 1);
        return NonMaximumSuppression(pixelList);
    }
}