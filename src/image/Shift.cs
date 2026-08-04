using System.Numerics;

namespace CVNet;

public static class CVShift
{
    // Optimized
    private static void shift<T>(
                CVImage imageIn, int xShift, int yShift,
                ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<T> bufferSpanIn = imageIn.BufferAs<T>();
        Span<T> bufferSpanOut = imageOut.BufferAs<T>();

        int planeSize = imageIn.Width * imageIn.Height;

        // When negative shifting, start at shifted position
        int startX = Math.Max(0, -xShift);
        int startY = Math.Max(0, -yShift);

        // When positive shifting, end before shifted position
        int endX = Math.Min(imageIn.Width, imageIn.Width - xShift);
        int endY = Math.Min(imageIn.Height, imageIn.Height - yShift);

        int copyWidth = endX - startX;

        for (int c = 0; c < imageIn.Channels; c++)
        {
            int colorOffset = c * planeSize;

            for (int y = startY; y < endY; y++)
            {
                int inRow = colorOffset + y * imageIn.Width + startX;
                int outRow = colorOffset + (y + yShift) * imageIn.Width + startX + xShift;

                bufferSpanIn.Slice(inRow, copyWidth).CopyTo(bufferSpanOut.Slice(outRow, copyWidth));
            }
        }
    }

    public static CVImage Shift(
                CVImage image,
                int xShift,
                int yShift)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) shift<byte>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) shift<sbyte>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) shift<ushort>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) shift<short>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) shift<uint>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) shift<int>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) shift<ulong>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) shift<long>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) shift<float>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) shift<double>(image, xShift, yShift, ref imageOut);

        return imageOut;
    }

    public static CVImage Shift<T>(
                    CVImage image,
                    int xShift,
                int yShift,
                    T[] defaultValue) where T : struct, INumber<T>
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats, defaultValue);

        if (image.DataFormat == CVDataFormat.CV_U8) shift<byte>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) shift<sbyte>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) shift<ushort>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) shift<short>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) shift<uint>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) shift<int>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) shift<ulong>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) shift<long>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) shift<float>(image, xShift, yShift, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) shift<double>(image, xShift, yShift, ref imageOut);

        return imageOut;
    }
}