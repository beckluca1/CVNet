using System.Numerics;
using System.Runtime.InteropServices;
using MathNet.Numerics.Providers.LinearAlgebra;

namespace CVNet;

public enum CVDataFormat
{
    CV_NONE,
    CV_U8,
    CV_U16,
    CV_U32,
    CV_U64,
    CV_S8,
    CV_S16,
    CV_S32,
    CV_S64,
    CV_F32,
    CV_F64,
}

public enum CVColorFormat
{
    CV_NONE,
    CV_C1,
    CV_C2,
    CV_C3,
    CV_C4,
}

public enum CVChannel
{
    CV_NONE,
    CV_R,
    CV_G,
    CV_B,
    CV_A,

    // Placeholders

    // Averages
    CV_AVG_RGB,
    CV_AVG_RGBA,

    // Fixed Values
    CV_A_ZERO,
    CV_A_ONE,
    CV_A_255,
}

public struct CVChannelFormat
{
    public CVChannel[] Channels = new CVChannel[] { CVChannel.CV_NONE };

    public CVChannelFormat()
    {
    }

    public static CVChannelFormat None()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_NONE } };
    }

    public static CVChannelFormat Grayscale()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_AVG_RGB } };
    }

    public static CVChannelFormat R()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_R } };
    }

    public static CVChannelFormat RGB()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B } };
    }

    public static CVChannelFormat RGBA()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B, CVChannel.CV_A } };
    }

    public static CVChannelFormat ARGB()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_A, CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B } };
    }

    public static CVChannelFormat BGR()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R } };
    }

    public static CVChannelFormat BGRA()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R, CVChannel.CV_A } };
    }

    public static CVChannelFormat ABGR()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_A, CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R } };
    }

    public static CVChannelFormat RRR255()
    {
        return new CVChannelFormat() { Channels = new CVChannel[] { CVChannel.CV_R, CVChannel.CV_R, CVChannel.CV_R, CVChannel.CV_A_255 } };
    }
}

public enum CV_ResizeMode
{
    CV_STRETCH_NEAREST,
    CV_STRETCH_LINEAR,
    CV_CROP_NEAREST,
    CV_CROP_LINEAR,
}

public class CVImage
{
    public int Width;
    public int Height;

    private int WidthHeight;
    private int ChannelsWidth;

    public int Channels;
    public int Bytes;

    public CVColorFormat ColorFormat;
    public CVDataFormat DataFormat;
    public CVChannelFormat ChannelFormat;

    public int bufferSize;
    public byte[] buffer;

    private CVImage(int width, int height, CVColorFormat colorFormat, CVDataFormat dataFormat, CVChannelFormat channelFormat)
    {
        Width = width;
        Height = height;

        Channels = (int)colorFormat;

        if (dataFormat == CVDataFormat.CV_NONE) Bytes = 0;
        else if (dataFormat == CVDataFormat.CV_U8) Bytes = 1;
        else if (dataFormat == CVDataFormat.CV_S8) Bytes = 1;
        else if (dataFormat == CVDataFormat.CV_U16) Bytes = 2;
        else if (dataFormat == CVDataFormat.CV_S16) Bytes = 2;
        else if (dataFormat == CVDataFormat.CV_U32) Bytes = 4;
        else if (dataFormat == CVDataFormat.CV_S32) Bytes = 4;
        else if (dataFormat == CVDataFormat.CV_U64) Bytes = 8;
        else if (dataFormat == CVDataFormat.CV_S64) Bytes = 8;
        else if (dataFormat == CVDataFormat.CV_F32) Bytes = 4;
        else if (dataFormat == CVDataFormat.CV_F64) Bytes = 8;

        WidthHeight = Width * Height;
        ChannelsWidth = Channels * Width;

        ColorFormat = colorFormat;
        DataFormat = dataFormat;
        ChannelFormat = channelFormat;

        bufferSize = Width * Height * Channels * Bytes;
        buffer = new byte[bufferSize];
    }

    public Span<T> BufferAs<T>() where T : struct
    {
        return MemoryMarshal.Cast<byte, T>(MemoryMarshal.AsBytes(buffer.AsSpan()));
    }

    private void InitPlanar<T>(T[] data)
    {
        Buffer.BlockCopy(data, 0, buffer, 0, bufferSize);
    }

    // Optimized
    private void InitInterleaved<T>(T[] data) where T : struct
    {
        Span<T> bufferOutSpan = BufferAs<T>();

        int pixelCount = WidthHeight;
        int channels = Channels;

        for (int c = 0; c < channels; c++)
        {
            int dstOffset = c * pixelCount;
            int srcOffset = c;

            int di = dstOffset;
            int si = srcOffset;

            for (int n = 0; n < pixelCount; n++)
            {
                bufferOutSpan[di] = data[si];
                di++;
                si += channels;
            }
        }
    }

    private void Init<T>(T[] data) where T : struct
    {
        Span<T> bufferSpan = BufferAs<T>();

        for (int i = 0; i < Channels; i++)
            bufferSpan.Slice(i * WidthHeight, WidthHeight).Fill(data[i]);
    }

    private void Init<T>(T data) where T : struct
    {
        Span<T> bufferSpan = BufferAs<T>();

        bufferSpan.Fill(data);
    }

    public static CVImage Create(int width = 0, int height = 0, CVColorFormat colorFormat = CVColorFormat.CV_NONE, CVDataFormat dataFormat = CVDataFormat.CV_NONE, CVChannelFormat channelFormat = default)
    {
        CVImage image = new CVImage(width, height, colorFormat, dataFormat, channelFormat);

        return image;
    }

    public static CVImage Create<T>(int width, int height, CVColorFormat colorFormat, CVDataFormat dataFormat, CVChannelFormat channelFormat, T data) where T : struct
    {
        CVImage image = new CVImage(width, height, colorFormat, dataFormat, channelFormat);

        image.Init(data);

        return image;
    }

    public static CVImage Create<T>(int width, int height, CVColorFormat colorFormat, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct
    {
        CVImage image = new CVImage(width, height, colorFormat, dataFormat, channelFormat);

        image.Init(data);

        return image;
    }

    public static CVImage CreatePlanar<T>(int width, int height, CVColorFormat colorFormat, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct
    {
        CVImage image = new CVImage(width, height, colorFormat, dataFormat, channelFormat);

        image.InitPlanar(data);

        return image;
    }

    public static CVImage CreateInterleaved<T>(int width, int height, CVColorFormat colorFormat, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct
    {
        CVImage image = new CVImage(width, height, colorFormat, dataFormat, channelFormat);

        image.InitInterleaved(data);

        return image;
    }

    public byte[] GetBufferPlanar()
    {
        return buffer;
    }

    public byte[] GetBufferInterleaved<T>(ref byte[] bufferOut) where T : struct
    {
        Span<T> bufferInSpan = BufferAs<T>();
        Span<T> bufferOutSpan = MemoryMarshal.Cast<byte, T>(MemoryMarshal.AsBytes(bufferOut.AsSpan()));

        int pixelCount = WidthHeight;
        int channels = Channels;

        for (int c = 0; c < channels; c++)
        {
            int dstOffset = c * pixelCount;
            int srcOffset = c;

            int di = dstOffset;
            int si = srcOffset;

            for (int n = 0; n < pixelCount; n++)
            {
                bufferOutSpan[si] = bufferInSpan[di];
                di++;
                si += channels;
            }
        }

        return bufferOut;
    }

    public byte[] GetBufferInterleaved()
    {
        byte[] bufferOut = new byte[buffer.Length];

        if (DataFormat == CVDataFormat.CV_U8) GetBufferInterleaved<byte>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_S8) GetBufferInterleaved<sbyte>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_U16) GetBufferInterleaved<ushort>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_S16) GetBufferInterleaved<short>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_U32) GetBufferInterleaved<uint>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_S32) GetBufferInterleaved<int>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_U64) GetBufferInterleaved<ulong>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_S64) GetBufferInterleaved<long>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_F32) GetBufferInterleaved<float>(ref bufferOut);
        else if (DataFormat == CVDataFormat.CV_F64) GetBufferInterleaved<double>(ref bufferOut);

        return bufferOut;
    }

    public static CVImage CreateSumMask(int width = 0, int height = 0, CVColorFormat colorFormat = CVColorFormat.CV_NONE, CVDataFormat dataFormat = CVDataFormat.CV_NONE)
    {
        CVImage image = Create(width, height, colorFormat, dataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) image.Init<byte>(1);
        else if (image.DataFormat == CVDataFormat.CV_S8) image.Init<sbyte>(1);
        else if (image.DataFormat == CVDataFormat.CV_U16) image.Init<ushort>(1);
        else if (image.DataFormat == CVDataFormat.CV_S16) image.Init<short>(1);
        else if (image.DataFormat == CVDataFormat.CV_U32) image.Init<uint>(1);
        else if (image.DataFormat == CVDataFormat.CV_S32) image.Init<int>(1);
        else if (image.DataFormat == CVDataFormat.CV_U64) image.Init<ulong>(1);
        else if (image.DataFormat == CVDataFormat.CV_S64) image.Init<long>(1);
        else if (image.DataFormat == CVDataFormat.CV_F32) image.Init<float>(1);
        else if (image.DataFormat == CVDataFormat.CV_F64) image.Init<double>(1);

        return image;
    }

    // Optimized
    public static void SetGaussianMask<T>(ref CVImage image) where T : struct, INumber<T>
    {
        Span<T> bufferSpan = image.BufferAs<T>();

        int rX = (image.Width - 1) / 2;
        int rY = (image.Height - 1) / 2;

        double invW = 1.0 / image.Width;
        double invH = 1.0 / image.Height;

        int planeSize = image.Width * image.Height;

        for (int c = 0; c < image.Channels; c++)
        {
            int cBase = c * planeSize;

            double sum = 0.0;

            for (int y = -rY; y <= rY; y++)
            {
                int yOff = (y + rY) * image.Width;

                double yCord = y * invH;
                double y2 = yCord * yCord;

                for (int x = -rX; x <= rX; x++)
                {
                    double xCord = x * invW;

                    double v = Math.Exp(-(xCord * xCord + y2));

                    int idx = cBase + yOff + (x + rX);

                    bufferSpan[idx] = (T)Convert.ChangeType(v, typeof(T));
                    sum += v;
                }
            }

            T sumInv = (T)Convert.ChangeType(sum, typeof(T));

            for (int y = -rY; y <= rY; y++)
            {
                int yOff = (y + rY) * image.Width;

                for (int x = -rX; x <= rX; x++)
                {
                    int idx = cBase + yOff + (x + rX);
                    bufferSpan[idx] /= sumInv;
                }
            }
        }
    }

    public static CVImage CreateGaussianMask(int width = 0, int height = 0, CVColorFormat colorFormat = CVColorFormat.CV_NONE, CVDataFormat dataFormat = CVDataFormat.CV_NONE)
    {
        CVImage image = Create(width, height, colorFormat, dataFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) SetGaussianMask<byte>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_S8) SetGaussianMask<sbyte>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_U16) SetGaussianMask<ushort>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_S16) SetGaussianMask<short>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_U32) SetGaussianMask<uint>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_S32) SetGaussianMask<int>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_U64) SetGaussianMask<ulong>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_S64) SetGaussianMask<long>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_F32) SetGaussianMask<float>(ref image);
        else if (image.DataFormat == CVDataFormat.CV_F64) SetGaussianMask<double>(ref image);

        return image;
    }

    public CVImage Clone()
    {
        return CreatePlanar(Width, Height, ColorFormat, DataFormat, ChannelFormat, buffer);
    }

    public static CVImage operator +(int val, CVImage image) { return CVMath.Add(image, val); }
    public static CVImage operator +(double val, CVImage image) { return CVMath.Add(image, val); }
    public static CVImage operator +(CVImage image, int val) { return CVMath.Add(image, val); }
    public static CVImage operator +(CVImage image, double val) { return CVMath.Add(image, val); }
    public static CVImage operator +(CVImage image1, CVImage image2) { return CVMath.AddUnsafe(image1, image2); }

    public static CVImage operator -(int val, CVImage image) { return CVMath.Subtract(val, image); }
    public static CVImage operator -(double val, CVImage image) { return CVMath.Subtract(val, image); }
    public static CVImage operator -(CVImage image, int val) { return CVMath.Subtract(image, val); }
    public static CVImage operator -(CVImage image, double val) { return CVMath.Subtract(image, val); }
    public static CVImage operator -(CVImage image1, CVImage image2) { return CVMath.Subtract(image1, image2); }

    public static CVImage operator *(int val, CVImage image) { return CVMath.Multiply(image, val); }
    public static CVImage operator *(double val, CVImage image) { return CVMath.Multiply(image, val); }
    public static CVImage operator *(CVImage image, int val) { return CVMath.Multiply(image, val); }
    public static CVImage operator *(CVImage image, double val) { return CVMath.Multiply(image, val); }
    public static CVImage operator *(CVImage image1, CVImage image2) { return CVMath.Multiply(image1, image2); }

    public static CVImage operator /(int val, CVImage image) { return CVMath.Divide(val, image); }
    public static CVImage operator /(double val, CVImage image) { return CVMath.Divide(val, image); }
    public static CVImage operator /(CVImage image, int val) { return CVMath.Divide(image, val); }
    public static CVImage operator /(CVImage image, double val) { return CVMath.Divide(image, val); }
    public static CVImage operator /(CVImage image1, CVImage image2) { return CVMath.Divide(image1, image2); }
}