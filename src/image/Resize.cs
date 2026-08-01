using System.Numerics;

namespace CVNet;

public static class CVResize
{
    private static void half<T>(CVImage imageIn, ref CVImage imageOut)
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

    public static CVImage Half(
            CVImage image)
    {
        CVImage imageOut = CVImage.Create(image.Width / 2, image.Height / 2, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) half<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) half<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) half<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) half<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) half<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) half<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) half<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) half<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) half<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) half<double>(image, ref imageOut);

        return imageOut;
    }

    private static void twice<T>(CVImage imageIn, ref CVImage imageOut)
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

    public static CVImage Twice(
            CVImage image)
    {
        CVImage imageOut = CVImage.Create(image.Width * 2, image.Height * 2, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) twice<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) twice<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) twice<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) twice<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) twice<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) twice<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) twice<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) twice<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) twice<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) twice<double>(image, ref imageOut);

        return imageOut;
    }

    // Optimized
    private static void stretchNearest<T>(
        CVImage imageIn,
        ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> bufferIn = imageIn.BufferAs<T>();
        Span<T> bufferOut = imageOut.BufferAs<T>();

        int inW = imageIn.Width;
        int inH = imageIn.Height;
        int outW = imageOut.Width;
        int outH = imageOut.Height;

        int channels = imageIn.Channels;
        int inPlane = inW * inH;
        int outPlane = outW * outH;

        int[] yMap = new int[outH];
        for (int y = 0; y < outH; y++)
            yMap[y] = y * inH / outH;

        int[] xMap = new int[outW];
        for (int x = 0; x < outW; x++)
            xMap[x] = x * inW / outW;

        for (int c = 0; c < channels; c++)
        {
            int inBase = c * inPlane;
            int outBase = c * outPlane;

            for (int y = 0; y < outH; y++)
            {
                int inRow = inBase + yMap[y] * inW;
                int outRow = outBase + y * outW;

                int inIndex = inRow;
                int outIndex = outRow;

                for (int x = 0; x < outW; x++)
                {
                    bufferOut[outIndex + x] = bufferIn[inIndex + xMap[x]];
                }
            }
        }
    }

    public static CVImage StretchNearest(
                CVImage image,
                int targetWidth,
                int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) stretchNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) stretchNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) stretchNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) stretchNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) stretchNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) stretchNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) stretchNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) stretchNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) stretchNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) stretchNearest<double>(image, ref imageOut);

        return imageOut;
    }

    // Optimized
    private static void stretchLinear<T>(
        CVImage imageIn,
        ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> bufferIn = imageIn.BufferAs<T>();
        Span<T> bufferOut = imageOut.BufferAs<T>();

        int inW = imageIn.Width;
        int inH = imageIn.Height;
        int outW = imageOut.Width;
        int outH = imageOut.Height;

        int channels = imageIn.Channels;
        int inPlane = inW * inH;
        int outPlane = outW * outH;

        // Precompute X interpolation
        int[] x0Map = new int[outW];
        int[] x1Map = new int[outW];
        double[] fxMap = new double[outW];

        double scaleX = (double)inW / outW;

        for (int x = 0; x < outW; x++)
        {
            double srcX = (x + 0.5) * scaleX - 0.5;

            int x0 = (int)Math.Floor(srcX);
            int x1 = x0 + 1;
            double fx = srcX - x0;

            if (x0 < 0)
            {
                x0 = 0;
                fx = 0;
            }

            if (x1 >= inW)
                x1 = inW - 1;

            x0Map[x] = x0;
            x1Map[x] = x1;
            fxMap[x] = fx;
        }

        // Precompute Y interpolation
        int[] y0Map = new int[outH];
        int[] y1Map = new int[outH];
        double[] fyMap = new double[outH];

        double scaleY = (double)inH / outH;

        for (int y = 0; y < outH; y++)
        {
            double srcY = (y + 0.5) * scaleY - 0.5;

            int y0 = (int)Math.Floor(srcY);
            int y1 = y0 + 1;
            double fy = srcY - y0;

            if (y0 < 0)
            {
                y0 = 0;
                fy = 0;
            }

            if (y1 >= inH)
                y1 = inH - 1;

            y0Map[y] = y0;
            y1Map[y] = y1;
            fyMap[y] = fy;
        }

        // Resize
        for (int c = 0; c < channels; c++)
        {
            int inBase = c * inPlane;
            int outBase = c * outPlane;

            for (int y = 0; y < outH; y++)
            {
                int y0 = y0Map[y];
                int y1 = y1Map[y];
                double fy = fyMap[y];

                int row0 = inBase + y0 * inW;
                int row1 = inBase + y1 * inW;

                int outRow = outBase + y * outW;

                double fy0 = 1.0 - fy;

                for (int x = 0; x < outW; x++)
                {
                    int x0 = x0Map[x];
                    int x1 = x1Map[x];
                    double fx = fxMap[x];

                    double fx0 = 1.0 - fx;

                    double value =
                        fx0 * fy0 * double.CreateChecked(bufferIn[row0 + x0]) +
                        fx * fy0 * double.CreateChecked(bufferIn[row0 + x1]) +
                        fx0 * fy * double.CreateChecked(bufferIn[row1 + x0]) +
                        fx * fy * double.CreateChecked(bufferIn[row1 + x1]);

                    bufferOut[outRow + x] = T.CreateChecked(value);
                }
            }
        }
    }

    public static CVImage StretchLinear(
            CVImage image,
            int targetWidth,
            int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) stretchLinear<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) stretchLinear<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) stretchLinear<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) stretchLinear<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) stretchLinear<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) stretchLinear<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) stretchLinear<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) stretchLinear<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) stretchLinear<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) stretchLinear<double>(image, ref imageOut);

        return imageOut;
    }

    // Optimized
    private static void crop<T>(
                CVImage imageIn,
                ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> bufferSpanIn = imageIn.BufferAs<T>();
        Span<T> bufferSpanOut = imageOut.BufferAs<T>();

        int dx = (imageIn.Width - imageOut.Width) / 2;
        int dy = (imageIn.Height - imageOut.Height) / 2;

        int srcX = Math.Max(0, dx);
        int srcY = Math.Max(0, dy);
        int dstX = Math.Max(0, -dx);
        int dstY = Math.Max(0, -dy);

        int copyWidth = Math.Min(imageOut.Width - dstX, imageIn.Width - srcX);
        int copyHeight = Math.Min(imageOut.Height - dstY, imageIn.Height - srcY);

        int inPlane = imageIn.Width * imageIn.Height;
        int outPlane = imageOut.Width * imageOut.Height;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int inBase = c * inPlane;
            int outBase = c * outPlane;

            for (int y = 0; y < copyHeight; y++)
            {
                int inRow = inBase + (srcY + y) * imageIn.Width + srcX;
                int outRow = outBase + (dstY + y) * imageOut.Width + dstX;

                bufferSpanIn.Slice(inRow, copyWidth).CopyTo(bufferSpanOut.Slice(outRow, copyWidth));
            }
        }
    }

    public static CVImage Crop(
                CVImage image,
                int targetWidth,
                int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) crop<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) crop<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) crop<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) crop<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) crop<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) crop<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) crop<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) crop<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) crop<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) crop<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage Crop<T>(
                    CVImage image,
                    int targetWidth,
                    int targetHeight,
                    T[] defaultValue) where T : struct, INumber<T>
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.DataFormat, image.ChannelFormats, defaultValue);

        if (image.DataFormat == CVDataFormat.CV_U8) crop<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) crop<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) crop<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) crop<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) crop<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) crop<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) crop<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) crop<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) crop<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) crop<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage Resize(CVImage image,
                                    int targetWidth,
                                    int targetHeight,
                                    CV_ResizeMode resizeMode)
    {
        if (resizeMode == CV_ResizeMode.CV_STRETCH_NEAREST) return StretchNearest(image, targetWidth, targetHeight);
        else if (resizeMode == CV_ResizeMode.CV_STRETCH_LINEAR) return StretchLinear(image, targetWidth, targetHeight);
        else if (resizeMode == CV_ResizeMode.CV_CROP) return Crop(image, targetWidth, targetHeight);

        throw new Exception("Unknown Resize mode");
    }
}