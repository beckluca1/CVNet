using System.Numerics;

namespace CVNet;

using VectorD = MathNet.Numerics.LinearAlgebra.Vector<double>;
using MatrixD = MathNet.Numerics.LinearAlgebra.Matrix<double>;
using DenseVectorD = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using DenseMatrixD = MathNet.Numerics.LinearAlgebra.Double.DenseMatrix;

public static class CVResizeTransformation
{
    private static void halfTransformation<T>(CVImage imageIn, ref CVImage imageOut)
    where T : struct, INumber<T>
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int outW = imageOut.Width;
        int outH = imageOut.Height;

        int dstAddRow = outW;
        int srcAddRow = outW * 4;

        int planeSize = outW * outH;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int srcPlane = c * planeSize * 4;
            int dstPlane = c * planeSize;

            int srcRow = srcPlane;
            int dstRow = dstPlane;

            for (int y = 0; y < outH; y++)
            {
                int s = srcRow;
                int d = dstRow;

                for (int x = 0; x < outW; x++)
                {
                    dst[d] = src[s];
                    d += 1;
                    s += 2;
                }

                srcRow += srcAddRow;
                dstRow += dstAddRow;
            }
        }
    }

    public static CVImage HalfTransformation(
            CVImage image)
    {
        CVImage imageOut = CVImage.Create(image.Width / 2, image.Height / 2, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) halfTransformation<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) halfTransformation<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) halfTransformation<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) halfTransformation<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) halfTransformation<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) halfTransformation<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) halfTransformation<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) halfTransformation<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) halfTransformation<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) halfTransformation<double>(image, ref imageOut);

        return imageOut;
    }

    private static void twiceTransformation<T>(CVImage imageIn, ref CVImage imageOut)
        where T : struct, INumber<T>
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int inW = imageIn.Width;
        int inH = imageIn.Height;

        int dstAddRow = inW * 4;
        int srcAddRow = inW;

        int planeSize = inW * inH;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int srcPlane = c * planeSize;
            int dstPlane = c * planeSize * 4;

            int srcRow = srcPlane;
            int dstRow = dstPlane;

            for (int y = 0; y < inH; y++)
            {
                int s = srcRow;
                int d = dstRow;

                for (int x = 0; x < inW; x++)
                {
                    dst[d] = src[s];
                    d += 2;
                    s += 1;
                }

                srcRow += srcAddRow;
                dstRow += dstAddRow;
            }
        }
    }

    public static CVImage TwiceTransformation(
            CVImage image)
    {
        CVImage imageOut = CVImage.Create(image.Width * 2, image.Height * 2, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) twiceTransformation<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) twiceTransformation<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) twiceTransformation<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) twiceTransformation<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) twiceTransformation<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) twiceTransformation<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) twiceTransformation<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) twiceTransformation<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) twiceTransformation<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) twiceTransformation<double>(image, ref imageOut);

        return imageOut;
    }

    // Optimized
    public static VectorD StretchTransformation(
        int currentWidth,
        int currentHeight,
        int targetWidth,
        int targetHeight,
        VectorD pointIn)
    {
        return DenseVectorD.OfArray([pointIn[0] * targetWidth / currentWidth, pointIn[1] * targetHeight / currentHeight]);
    }

    // Optimized
    public static VectorD CropTransformation(
        int currentWidth,
        int currentHeight,
        int targetWidth,
        int targetHeight,
        VectorD pointIn)
    {
        double dx = (currentWidth - targetWidth) / 2.0;
        double dy = (currentHeight - targetHeight) / 2.0;

        return DenseVectorD.OfArray([pointIn[0] - dx, pointIn[1] - dy]);
    }

    public static VectorD AspectCropTransformation(
        int currentWidth,
        int currentHeight,
        double aspect,
        VectorD pointIn,
        out int width,
        out int height)
    {
        width = (int)(currentHeight * aspect);
        height = currentHeight;

        if (width > currentWidth)
        {
            width = currentWidth;
            height = (int)(currentWidth / aspect);
        }

        return CropTransformation(currentWidth, currentHeight, width, height, pointIn);
    }

    public static VectorD AspectPadTransformation(
        int currentWidth,
        int currentHeight,
        double aspect,
        VectorD pointIn,
        out int width,
        out int height)
    {
        width = (int)(currentHeight * aspect);
        height = currentHeight;

        if (width < currentWidth)
        {
            width = currentWidth;
            height = (int)(currentWidth / aspect);
        }

        return CropTransformation(currentWidth, currentHeight, width, height, pointIn);
    }

    public static VectorD AspectFitWidthTransformation(
        int currentWidth,
        int currentHeight,
        double aspect,
        VectorD pointIn,
        out int width,
        out int height)
    {
        width = (int)(currentHeight * aspect);
        height = currentHeight;

        return StretchTransformation(currentWidth, currentHeight, width, height, pointIn);
    }

    public static VectorD AspectFitHeightTransformation(
        int currentWidth,
        int currentHeight,
        double aspect,
        VectorD pointIn,
        out int width,
        out int height)
    {
        width = currentWidth;
        height = (int)(currentWidth / aspect);

        return StretchTransformation(currentWidth, currentHeight, width, height, pointIn);
    }

    public static VectorD ResizeStretchTransformation(
        int currentWidth,
        int currentHeight,
        int width,
        int height,
        VectorD pointIn)
    {
        return StretchTransformation(currentWidth, currentHeight, width, height, pointIn);
    }

    public static VectorD ResizeCropTransformation(
        int currentWidth,
        int currentHeight,
        int width,
        int height,
        VectorD pointIn)
    {
        double aspect = (double)width / height;

        VectorD cropped = AspectCropTransformation(currentWidth, currentHeight, aspect, pointIn, out int croppedWidth, out int croppedHeight);
        return StretchTransformation(croppedWidth, croppedHeight, width, height, cropped);
    }

    public static VectorD ResizePadTransformation(
            int currentWidth,
            int currentHeight,
            int width,
            int height,
            VectorD pointIn)
    {
        double aspect = (double)width / height;

        VectorD padded = AspectPadTransformation(currentWidth, currentHeight, aspect, pointIn, out int paddedWidth, out int paddedHeight);
        return StretchTransformation(paddedWidth, paddedHeight, width, height, padded);
    }

    public static VectorD ResizeTransformation(
            int currentWidth,
            int currentHeight,
            int width,
            int height,
            VectorD pointIn,
            CVResizeMode resizeMode)
    {
        if (resizeMode == CVResizeMode.CV_STRETCH) return ResizeStretchTransformation(currentWidth, currentHeight, width, height, pointIn);
        else if (resizeMode == CVResizeMode.CV_CROP) return ResizeCropTransformation(currentWidth, currentHeight, width, height, pointIn);
        else if (resizeMode == CVResizeMode.CV_PAD) return ResizePadTransformation(currentWidth, currentHeight, width, height, pointIn);

        throw new Exception("Unknown Resize mode");
    }

    public static VectorD ResizeStretchTransformation(
        CVStretchTransform transform,
        VectorD pointIn)
    {
        return ResizeStretchTransformation(transform.SourceWidth, transform.SourceHeight, transform.TargetWidth, transform.TargetHeight, pointIn);
    }

    public static VectorD ResizeCropTransformation(
        CVCropTransform transform,
        VectorD pointIn)
    {
        return ResizeCropTransformation(transform.SourceWidth, transform.SourceHeight, transform.TargetWidth, transform.TargetHeight, pointIn);
    }

    public static VectorD ResizePadTransformation(
        CVPadTransform transform,
        VectorD pointIn)
    {
        return ResizePadTransformation(transform.SourceWidth, transform.SourceHeight, transform.TargetWidth, transform.TargetHeight, pointIn);
    }

    public static VectorD ResizeTransformation(
            CVRectTransform transform,
            VectorD pointIn)
    {
        if (transform is CVStretchTransform stretchTransform) return ResizeStretchTransformation(stretchTransform, pointIn);
        else if (transform is CVCropTransform cropTransform) return ResizeCropTransformation(cropTransform, pointIn);
        else if (transform is CVPadTransform padTransform) return ResizePadTransformation(padTransform, pointIn);

        throw new Exception("Unknown Resize mode");
    }
}