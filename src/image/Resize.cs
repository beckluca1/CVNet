namespace CVNet;

public static class CVResize
{
    // Optimized
    private static void StretchNearest<T>(
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

    private static CVImage StretchNearest<T>(
            CVImage image,
            int targetWidth,
            int targetHeight,
            T[] defaultValue) where T : struct
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, defaultValue);

        if (image.DataFormat == CVDataFormat.CV_U8) StretchNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) StretchNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) StretchNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) StretchNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) StretchNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) StretchNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) StretchNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) StretchNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) StretchNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) StretchNearest<double>(image, ref imageOut);

        return imageOut;
    }

    // Optimized
    private static void CropNearest<T>(
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

    private static CVImage CropNearest<T>(
                CVImage image,
                int targetWidth,
                int targetHeight,
                T[] defaultValue) where T : struct
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat, defaultValue);

        if (image.DataFormat == CVDataFormat.CV_U8) CropNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) CropNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) CropNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) CropNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) CropNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) CropNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) CropNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) CropNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) CropNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) CropNearest<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage ResizeImage<T>(CVImage image,
                                    int targetWidth,
                                    int targetHeight,
                                    CV_ResizeMode resizeMode,
                                    T[] defaultValue) where T : struct
    {
        if (resizeMode == CV_ResizeMode.CV_STRETCH) return StretchNearest(image, targetWidth, targetHeight, defaultValue);
        else if (resizeMode == CV_ResizeMode.CV_CROP) return CropNearest(image, targetWidth, targetHeight, defaultValue);

        throw new Exception("Unknown Resize mode");
    }

    private static CVImage StretchNearest(
            CVImage image,
            int targetWidth,
            int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) StretchNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) StretchNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) StretchNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) StretchNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) StretchNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) StretchNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) StretchNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) StretchNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) StretchNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) StretchNearest<double>(image, ref imageOut);

        return imageOut;
    }

    private static CVImage CropNearest(
                CVImage image,
                int targetWidth,
                int targetHeight)
    {
        CVImage imageOut = CVImage.Create(targetWidth, targetHeight, image.ColorFormat, image.DataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) CropNearest<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) CropNearest<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) CropNearest<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) CropNearest<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) CropNearest<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) CropNearest<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) CropNearest<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) CropNearest<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) CropNearest<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) CropNearest<double>(image, ref imageOut);

        return imageOut;
    }

    public static CVImage ResizeImage(CVImage image,
                                    int targetWidth,
                                    int targetHeight,
                                    CV_ResizeMode resizeMode)
    {
        if (resizeMode == CV_ResizeMode.CV_STRETCH) return StretchNearest(image, targetWidth, targetHeight);
        else if (resizeMode == CV_ResizeMode.CV_CROP) return CropNearest(image, targetWidth, targetHeight);

        throw new Exception("Unknown Resize mode");
    }
}