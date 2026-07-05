namespace CVNet;

public static class CVResize
{
    private static void half<T>(CVImage imageIn, ref CVImage imageOut)
    where T : struct
    {
        Span<T> src = imageIn.BufferAs<T>();
        Span<T> dst = imageOut.BufferAs<T>();

        int outW = imageOut.Width;
        int outH = imageOut.Height;

        int srcDRow = outW * 4;

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
                    dst[d++] = src[s];
                    s += 2;
                }

                srcRow += srcDRow;
                dstRow += outW;
            }
        }
    }

    public static CVImage Half<T>(
            CVImage image,
            T[] defaultValue) where T : struct
    {
        CVImage imageOut = CVImage.Create(image.Width / 2, image.Height / 2, image.ColorFormat, image.DataFormat, image.ChannelFormat, defaultValue);

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

    // Optimized
    private static void stretchNearest<T>(
        CVImage imageIn,
        ref CVImage imageOut) where T : struct
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

        // allocate once per call (still OK), but avoid per-pixel indexing
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

                // faster inner loop (no extra additions per iteration)
                for (int x = 0; x < outW; x++)
                {
                    bufferOut[outIndex + x] = bufferIn[inIndex + xMap[x]];
                }
            }
        }
    }

    public static CVImage StretchNearest<T>(
            CVImage image,
            int targetWidth,
            int targetHeight,
            T[] defaultValue) where T : struct
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, image.ChannelFormat, defaultValue);

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
    private static void cropNearest<T>(
                CVImage imageIn,
                ref CVImage imageOut) where T : struct
    {
        Span<T> bufferSpanIn = imageIn.BufferAs<T>();
        Span<T> bufferSpanOut = imageOut.BufferAs<T>();

        int dx = (imageIn.Width - imageOut.Width) / 2;
        int dy = (imageIn.Height - imageOut.Height) / 2;

        int inPlane = imageIn.Width * imageIn.Height;
        int outPlane = imageOut.Width * imageOut.Height;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int inBase = c * inPlane;
            int outBase = c * outPlane;

            for (int y = 0; y < imageOut.Height; y++)
            {
                int inY = y + dy;

                if ((uint)inY >= (uint)imageIn.Height)
                    continue;

                int inRow = inBase + inY * imageIn.Width;
                int outRow = outBase + y * imageOut.Width;

                for (int x = 0; x < imageOut.Width; x++)
                {
                    int inX = x + dx;

                    if ((uint)inX < (uint)imageIn.Width)
                        bufferSpanOut[outRow + x] = bufferSpanIn[inRow + inX];
                }
            }
        }
    }

    public static CVImage CropNearest<T>(
                CVImage image,
                int targetWidth,
                int targetHeight,
                T[] defaultValue) where T : struct
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, image.ChannelFormat, defaultValue);

        if (image.DataFormat == CVDataFormat.CV_U8) cropNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) cropNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) cropNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) cropNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) cropNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) cropNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) cropNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) cropNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) cropNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) cropNearest<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage StretchNearest(
            CVImage image,
            int targetWidth,
            int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, image.ChannelFormat);

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

    public static CVImage CropNearest(
                CVImage image,
                int targetWidth,
                int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, image.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) cropNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) cropNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) cropNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) cropNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) cropNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) cropNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) cropNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) cropNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) cropNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) cropNearest<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage ResizeImage<T>(CVImage image,
                                    int targetWidth,
                                    int targetHeight,
                                    CV_ResizeMode resizeMode,
                                    T[] defaultValue) where T : struct
    {
        if (resizeMode == CV_ResizeMode.CV_STRETCH_NEAREST) return StretchNearest(image, targetWidth, targetHeight, defaultValue);
        else if (resizeMode == CV_ResizeMode.CV_CROP_NEAREST) return CropNearest(image, targetWidth, targetHeight, defaultValue);

        throw new Exception("Unknown Resize mode");
    }

    public static CVImage ResizeImage(CVImage image,
                                    int targetWidth,
                                    int targetHeight,
                                    CV_ResizeMode resizeMode)
    {
        if (resizeMode == CV_ResizeMode.CV_STRETCH_NEAREST) return StretchNearest(image, targetWidth, targetHeight);
        else if (resizeMode == CV_ResizeMode.CV_CROP_NEAREST) return CropNearest(image, targetWidth, targetHeight);

        throw new Exception("Unknown Resize mode");
    }
}