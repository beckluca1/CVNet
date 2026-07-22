using System.Numerics;

namespace CVNet;

public class CVWindowing
{
    private static void integralImage<T>(
            CVImage image1,
            ref CVImage imageOut)
            where T : struct, INumber<T>
    {
        Span<T> src = image1.BufferAs<T>();
        Span<double> dst = imageOut.BufferAs<double>();

        int srcWidth = image1.Width;
        int srcHeight = image1.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcPlaneSize = srcWidth * srcHeight;
        int dstPlaneSize = dstWidth * dstHeight;

        for (int c = 0; c < image1.Channels; c++)
        {
            int srcChannelOffset = c * srcPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;

            for (int y = 0; y < srcHeight; y++)
            {
                int srcRow = srcChannelOffset + y * srcWidth;

                // integral image has +1 border
                int dstPrevRow = dstChannelOffset + y * dstWidth;
                int dstCurrRow = dstPrevRow + dstWidth;

                double rowSum = 0.0;

                for (int x = 0; x < srcWidth; x++)
                {
                    rowSum += double.CreateChecked(src[srcRow + x]);
                    dst[dstCurrRow + x + 1] = dst[dstPrevRow + x + 1] + rowSum;
                }
            }
        }
    }

    public static CVImage IntegralImage(CVImage image)
    {
        CVImage outImage = CVImage.Create(image.Width + 1, image.Height + 1, CVDataFormat.CV_F64, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) integralImage<byte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) integralImage<sbyte>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) integralImage<ushort>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) integralImage<short>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) integralImage<uint>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) integralImage<int>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) integralImage<ulong>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) integralImage<long>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) integralImage<float>(image, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) integralImage<double>(image, ref outImage);

        return outImage;
    }

    private static void sumWindow<T>(
    CVImage integralImage,
    int radius,
    ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<double> src = integralImage.BufferAs<double>();
        Span<T> dst = imageOut.BufferAs<T>();

        int srcWidth = integralImage.Width;
        int srcHeight = integralImage.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int srcPlaneSize = srcWidth * srcHeight;
        int dstPlaneSize = dstWidth * dstHeight;

        // Precompute x bounds
        int[] left = new int[dstWidth];
        int[] right = new int[dstWidth];

        for (int x = 0; x < dstWidth; x++)
        {
            left[x] = Math.Max(0, x - radius);
            right[x] = Math.Min(dstWidth - 1, x + radius) + 1;
        }

        int[] top = new int[dstHeight];
        int[] bottom = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            top[y] = Math.Max(0, y - radius);
            bottom[y] = Math.Min(dstHeight - 1, y + radius) + 1;
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * srcPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int row0 = srcChannelOffset + top[y] * srcWidth;
                int row1 = srcChannelOffset + bottom[y] * srcWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int x0 = left[x];
                    int x1 = right[x];

                    dst[dstRow + x] = T.CreateChecked(
                        src[row1 + x1]
                      - src[row0 + x1]
                      - src[row1 + x0]
                      + src[row0 + x0]);
                }
            }
        }
    }

    public static CVImage SumWindow(CVImage image, int radius)
    {
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) sumWindow<byte>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) sumWindow<sbyte>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) sumWindow<ushort>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) sumWindow<short>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) sumWindow<uint>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) sumWindow<int>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) sumWindow<ulong>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) sumWindow<long>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) sumWindow<float>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) sumWindow<double>(integralImage, radius, ref outImage);

        return outImage;
    }

    private static void averageWindow<T>(
        CVImage integralImage,
        int radius,
        ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<double> src = integralImage.BufferAs<double>();
        Span<T> dst = imageOut.BufferAs<T>();

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int iiPlaneSize = iiWidth * iiHeight;
        int dstPlaneSize = dstWidth * dstHeight;

        // Precompute x bounds
        int[] left = new int[dstWidth];
        int[] right = new int[dstWidth];

        for (int x = 0; x < dstWidth; x++)
        {
            left[x] = Math.Max(0, x - radius);
            right[x] = Math.Min(iiWidth - 1, x + radius + 1);
        }

        int[] top = new int[dstHeight];
        int[] bottom = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            top[y] = Math.Max(0, y - radius);
            bottom[y] = Math.Min(iiHeight - 1, y + radius + 1);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int iiChannelOffset = c * iiPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int windowHeight = bottom[y] - top[y];

                int row0 = iiChannelOffset + top[y] * iiWidth;
                int row1 = iiChannelOffset + bottom[y] * iiWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int x0 = left[x];
                    int x1 = right[x];

                    int windowWidth = x1 - x0;
                    int area = windowWidth * windowHeight;

                    if (area == 0) continue;

                    dst[dstRow + x] = T.CreateChecked(
                        (src[row1 + x1]
                      - src[row0 + x1]
                      - src[row1 + x0]
                      + src[row0 + x0]) / area);
                }
            }
        }
    }

    public static CVImage AverageWindow(CVImage image, int radius)
    {
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(image.Width, image.Height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) averageWindow<byte>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) averageWindow<sbyte>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) averageWindow<ushort>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) averageWindow<short>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) averageWindow<uint>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) averageWindow<int>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) averageWindow<ulong>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) averageWindow<long>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) averageWindow<float>(integralImage, radius, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) averageWindow<double>(integralImage, radius, ref outImage);

        return outImage;
    }

    private static void sumWindowResample<T>(
            CVImage integralImage,
            ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<double> src = integralImage.BufferAs<double>();
        Span<T> dst = imageOut.BufferAs<T>();

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int srcWidth = iiWidth - 1;
        int srcHeight = iiHeight - 1;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int iiPlaneSize = iiWidth * iiHeight;
        int dstPlaneSize = dstWidth * dstHeight;

        int[] left = new int[dstWidth];
        int[] right = new int[dstWidth];

        double xRatio = (double)srcWidth / dstWidth;
        double yRatio = (double)srcHeight / dstHeight;

        for (int x = 0; x < dstWidth; x++)
        {
            left[x] = Math.Clamp((int)(x * xRatio), 0, srcWidth);
            right[x] = Math.Clamp((int)((x + 1) * xRatio), 0, srcWidth);
        }

        int[] top = new int[dstHeight];
        int[] bottom = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            top[y] = Math.Clamp((int)(y * yRatio), 0, srcHeight);
            bottom[y] = Math.Clamp((int)((y + 1) * yRatio), 0, srcHeight);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * iiPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int row0 = srcChannelOffset + top[y] * iiWidth;
                int row1 = srcChannelOffset + bottom[y] * iiWidth;

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int x0 = left[x];
                    int x1 = right[x];

                    dst[dstRow + x] = T.CreateChecked(
                        src[row1 + x1]
                      - src[row0 + x1]
                      - src[row1 + x0]
                      + src[row0 + x0]);
                }
            }
        }
    }

    public static CVImage SumWindowResample(CVImage image, int width, int height)
    {
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(width, height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) sumWindowResample<byte>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) sumWindowResample<sbyte>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) sumWindowResample<ushort>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) sumWindowResample<short>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) sumWindowResample<uint>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) sumWindowResample<int>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) sumWindowResample<ulong>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) sumWindowResample<long>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) sumWindowResample<float>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) sumWindowResample<double>(integralImage, ref outImage);

        return outImage;
    }

    private static void averageWindowResample<T>(
                CVImage integralImage,
                ref CVImage imageOut) where T : struct, INumber<T>
    {
        Span<double> src = integralImage.BufferAs<double>();
        Span<T> dst = imageOut.BufferAs<T>();

        int iiWidth = integralImage.Width;
        int iiHeight = integralImage.Height;

        int srcWidth = iiWidth - 1;
        int srcHeight = iiHeight - 1;

        int dstWidth = imageOut.Width;
        int dstHeight = imageOut.Height;

        int iiPlaneSize = iiWidth * iiHeight;
        int dstPlaneSize = dstWidth * dstHeight;

        int[] left = new int[dstWidth];
        int[] right = new int[dstWidth];

        double xRatio = (double)srcWidth / dstWidth;
        double yRatio = (double)srcHeight / dstHeight;

        for (int x = 0; x < dstWidth; x++)
        {
            left[x] = Math.Clamp((int)(x * xRatio), 0, srcWidth);
            right[x] = Math.Clamp((int)((x + 1) * xRatio), 0, srcWidth);
        }

        int[] top = new int[dstHeight];
        int[] bottom = new int[dstHeight];

        for (int y = 0; y < dstHeight; y++)
        {
            top[y] = Math.Clamp((int)(y * yRatio), 0, srcHeight);
            bottom[y] = Math.Clamp((int)((y + 1) * yRatio), 0, srcHeight);
        }

        for (int c = 0; c < integralImage.Channels; c++)
        {
            int srcChannelOffset = c * iiPlaneSize;
            int dstChannelOffset = c * dstPlaneSize;

            for (int y = 0; y < dstHeight; y++)
            {
                int row0 = srcChannelOffset + top[y] * iiWidth;
                int row1 = srcChannelOffset + bottom[y] * iiWidth;

                int windowHeight = bottom[y] - top[y];

                int dstRow = dstChannelOffset + y * dstWidth;

                for (int x = 0; x < dstWidth; x++)
                {
                    int x0 = left[x];
                    int x1 = right[x];

                    int windowWidth = x1 - x0;
                    int area = windowWidth * windowHeight;

                    if (area == 0) continue;

                    dst[dstRow + x] = T.CreateChecked(
                        (src[row1 + x1]
                      - src[row0 + x1]
                      - src[row1 + x0]
                      + src[row0 + x0]) / area);
                }
            }
        }
    }

    public static CVImage AverageWindowResample(CVImage image, int width, int height)
    {
        CVImage integralImage = IntegralImage(image);
        CVImage outImage = CVImage.Create(width, height, image.DataFormat, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) averageWindowResample<byte>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) averageWindowResample<sbyte>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) averageWindowResample<ushort>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) averageWindowResample<short>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) averageWindowResample<uint>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) averageWindowResample<int>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) averageWindowResample<ulong>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) averageWindowResample<long>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) averageWindowResample<float>(integralImage, ref outImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) averageWindowResample<double>(integralImage, ref outImage);

        return outImage;
    }
}