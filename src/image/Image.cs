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

    // Transforms
    CV_G_FROM_R,
    CV_B_FROM_R,
    CV_A_FROM_R,
}

public enum CVChannelFormat
{
    CV_None,
    CV_Grayscale,
    CV_RRR,
    CV_RRR255,
    CV_R,
    CV_RGB,
    CV_RGBA,
    CV_ARGB,
    CV_BGR,
    CV_BGRA,
    CV_ABGR,
}

public struct CVChannelFormats
{
    public CVChannel[] Channels = [CVChannel.CV_NONE];

    public CVChannelFormats()
    {
        Channels = [CVChannel.CV_NONE];
    }

    public CVChannelFormats(CVChannel[] channels)
    {
        if (channels == null) throw new Exception("Channels are NULL");

        Channels = (CVChannel[])channels.Clone();
    }

    public CVChannelFormats Clone()
    {
        return new CVChannelFormats(Channels);
    }

    public CVChannelFormats(CVChannelFormat id)
    {
        if (id == CVChannelFormat.CV_None) Channels = [CVChannel.CV_NONE];
        else if (id == CVChannelFormat.CV_Grayscale) Channels = [CVChannel.CV_AVG_RGB];
        else if (id == CVChannelFormat.CV_RRR) Channels = [CVChannel.CV_R, CVChannel.CV_G_FROM_R, CVChannel.CV_B_FROM_R];
        else if (id == CVChannelFormat.CV_RRR255) Channels = [CVChannel.CV_R, CVChannel.CV_G_FROM_R, CVChannel.CV_B_FROM_R, CVChannel.CV_A_255];
        else if (id == CVChannelFormat.CV_R) Channels = [CVChannel.CV_R];
        else if (id == CVChannelFormat.CV_RGB) Channels = [CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B];
        else if (id == CVChannelFormat.CV_RGBA) Channels = [CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B, CVChannel.CV_A];
        else if (id == CVChannelFormat.CV_ARGB) Channels = [CVChannel.CV_A, CVChannel.CV_R, CVChannel.CV_G, CVChannel.CV_B];
        else if (id == CVChannelFormat.CV_BGR) Channels = [CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R];
        else if (id == CVChannelFormat.CV_BGRA) Channels = [CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R, CVChannel.CV_A];
        else if (id == CVChannelFormat.CV_ABGR) Channels = [CVChannel.CV_A, CVChannel.CV_B, CVChannel.CV_G, CVChannel.CV_R];
    }
}

public enum CV_ResizeMode
{
    CV_HALF,
    CV_DOUBLE,
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

    public CVDataFormat DataFormat;
    public CVChannelFormats ChannelFormats;

    public int bufferSize;
    public byte[] buffer;

    private CVImage(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats)
    {
        Width = width;
        Height = height;

        DataFormat = dataFormat;
        // Clone here because ChannelFormats contains array
        ChannelFormats = channelFormats.Clone();

        if (DataFormat == CVDataFormat.CV_NONE) Bytes = 0;
        else if (DataFormat == CVDataFormat.CV_U8) Bytes = 1;
        else if (DataFormat == CVDataFormat.CV_S8) Bytes = 1;
        else if (DataFormat == CVDataFormat.CV_U16) Bytes = 2;
        else if (DataFormat == CVDataFormat.CV_S16) Bytes = 2;
        else if (DataFormat == CVDataFormat.CV_U32) Bytes = 4;
        else if (DataFormat == CVDataFormat.CV_S32) Bytes = 4;
        else if (DataFormat == CVDataFormat.CV_U64) Bytes = 8;
        else if (DataFormat == CVDataFormat.CV_S64) Bytes = 8;
        else if (DataFormat == CVDataFormat.CV_F32) Bytes = 4;
        else if (DataFormat == CVDataFormat.CV_F64) Bytes = 8;
        Channels = ChannelFormats.Channels.Length;

        WidthHeight = Width * Height;
        ChannelsWidth = Channels * Width;

        bufferSize = Width * Height * Channels * Bytes;
        buffer = new byte[bufferSize];
    }

    public Span<T> BufferAs<T>() where T : struct, INumber<T>
    {
        return MemoryMarshal.Cast<byte, T>(MemoryMarshal.AsBytes(buffer.AsSpan()));
    }

    public Span<T> ChannelAs<T>(int channel) where T : struct, INumber<T>
    {
        return BufferAs<T>().Slice(channel * WidthHeight, WidthHeight);
    }

    private void InitPlanar<T>(T[] data)
    {
        Buffer.BlockCopy(data, 0, buffer, 0, bufferSize);
    }

    // Optimized
    private void InitInterleaved<T>(T[] data) where T : struct, INumber<T>
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

    private void InitSafe<T, TV>(TV[] data) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        Span<T> bufferSpan = BufferAs<T>();

        for (int i = 0; i < Channels; i++)
        {
            T dataConv = T.CreateChecked(data[i]);
            bufferSpan.Slice(i * WidthHeight, WidthHeight).Fill(dataConv);
        }
    }

    private void InitSafe<T>(T[] data) where T : struct, INumber<T>
    {
        if (DataFormat == CVDataFormat.CV_U8) InitSafe<byte, T>(data);
        else if (DataFormat == CVDataFormat.CV_S8) InitSafe<sbyte, T>(data);
        else if (DataFormat == CVDataFormat.CV_U16) InitSafe<ushort, T>(data);
        else if (DataFormat == CVDataFormat.CV_S16) InitSafe<short, T>(data);
        else if (DataFormat == CVDataFormat.CV_U32) InitSafe<uint, T>(data);
        else if (DataFormat == CVDataFormat.CV_S32) InitSafe<int, T>(data);
        else if (DataFormat == CVDataFormat.CV_U64) InitSafe<ulong, T>(data);
        else if (DataFormat == CVDataFormat.CV_S64) InitSafe<long, T>(data);
        else if (DataFormat == CVDataFormat.CV_F32) InitSafe<float, T>(data);
        else if (DataFormat == CVDataFormat.CV_F64) InitSafe<double, T>(data);
    }

    private void InitUnsafe<T>(T[] data) where T : struct, INumber<T>
    {
        Span<T> bufferSpan = BufferAs<T>();

        for (int i = 0; i < Channels; i++)
        {
            bufferSpan.Slice(i * WidthHeight, WidthHeight).Fill(data[i]);
        }
    }

    private void InitSafe<T, TV>(TV data) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        T dataConv = T.CreateChecked(data);

        Span<T> bufferSpan = BufferAs<T>();

        bufferSpan.Fill(dataConv);
    }

    private void InitSafe<T>(T data) where T : struct, INumber<T>
    {
        if (DataFormat == CVDataFormat.CV_U8) InitSafe<byte, T>(data);
        else if (DataFormat == CVDataFormat.CV_S8) InitSafe<sbyte, T>(data);
        else if (DataFormat == CVDataFormat.CV_U16) InitSafe<ushort, T>(data);
        else if (DataFormat == CVDataFormat.CV_S16) InitSafe<short, T>(data);
        else if (DataFormat == CVDataFormat.CV_U32) InitSafe<uint, T>(data);
        else if (DataFormat == CVDataFormat.CV_S32) InitSafe<int, T>(data);
        else if (DataFormat == CVDataFormat.CV_U64) InitSafe<ulong, T>(data);
        else if (DataFormat == CVDataFormat.CV_S64) InitSafe<long, T>(data);
        else if (DataFormat == CVDataFormat.CV_F32) InitSafe<float, T>(data);
        else if (DataFormat == CVDataFormat.CV_F64) InitSafe<double, T>(data);
    }

    private void InitUnsafe<T>(T data) where T : struct, INumber<T>
    {
        Span<T> bufferSpan = BufferAs<T>();

        bufferSpan.Fill(data);
    }

    public static CVImage Create(int width = 0, int height = 0, CVDataFormat dataFormat = CVDataFormat.CV_NONE, CVChannelFormat channelFormat = CVChannelFormat.CV_None)
    {
        CVChannelFormats channelFormats = new CVChannelFormats(channelFormat);

        return Create(width, height, dataFormat, channelFormats);
    }

    public static CVImage Create(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats)
    {
        CVImage image = new CVImage(width, height, dataFormat, channelFormats);

        return image;
    }

    public static CVImage Create<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormat channelFormat, T data) where T : struct, INumber<T>
    {
        return Create<T>(width, height, dataFormat, new CVChannelFormats(channelFormat), data);
    }

    public static CVImage Create<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats, T data) where T : struct, INumber<T>
    {
        CVImage image = new CVImage(width, height, dataFormat, channelFormats);

        image.InitSafe(data);

        return image;
    }

    public static CVImage Create<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct, INumber<T>
    {
        return Create<T>(width, height, dataFormat, new CVChannelFormats(channelFormat), data);
    }

    public static CVImage Create<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats, T[] data) where T : struct, INumber<T>
    {
        CVImage image = new CVImage(width, height, dataFormat, channelFormats);

        image.InitSafe(data);

        return image;
    }

    public static CVImage CreatePlanar<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct, INumber<T>
    {
        return CreatePlanar<T>(width, height, dataFormat, new CVChannelFormats(channelFormat), data);
    }

    public static CVImage CreatePlanar<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats, T[] data) where T : struct, INumber<T>
    {
        CVImage image = new CVImage(width, height, dataFormat, channelFormats);

        image.InitPlanar(data);

        return image;
    }

    public static CVImage CreateInterleaved<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormat channelFormat, T[] data) where T : struct, INumber<T>
    {
        return CreateInterleaved<T>(width, height, dataFormat, new CVChannelFormats(channelFormat), data);
    }

    public static CVImage CreateInterleaved<T>(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats, T[] data) where T : struct, INumber<T>
    {
        CVImage image = new CVImage(width, height, dataFormat, channelFormats);

        image.InitInterleaved(data);

        return image;
    }

    public byte[] GetBufferPlanar()
    {
        return buffer;
    }

    public byte[] GetBufferInterleaved<T>(ref byte[] bufferOut) where T : struct, INumber<T>
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

    public static CVImage CreateSumMask(int width = 0, int height = 0, CVDataFormat dataFormat = CVDataFormat.CV_NONE, CVChannelFormat channelFormat = CVChannelFormat.CV_None)
    {
        return CreateSumMask(width, height, dataFormat, new CVChannelFormats(channelFormat));
    }

    public static CVImage CreateSumMask(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats)
    {
        CVImage image = Create(width, height, dataFormat, channelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) image.InitUnsafe<byte>(1);
        else if (image.DataFormat == CVDataFormat.CV_S8) image.InitUnsafe<sbyte>(1);
        else if (image.DataFormat == CVDataFormat.CV_U16) image.InitUnsafe<ushort>(1);
        else if (image.DataFormat == CVDataFormat.CV_S16) image.InitUnsafe<short>(1);
        else if (image.DataFormat == CVDataFormat.CV_U32) image.InitUnsafe<uint>(1);
        else if (image.DataFormat == CVDataFormat.CV_S32) image.InitUnsafe<int>(1);
        else if (image.DataFormat == CVDataFormat.CV_U64) image.InitUnsafe<ulong>(1);
        else if (image.DataFormat == CVDataFormat.CV_S64) image.InitUnsafe<long>(1);
        else if (image.DataFormat == CVDataFormat.CV_F32) image.InitUnsafe<float>(1);
        else if (image.DataFormat == CVDataFormat.CV_F64) image.InitUnsafe<double>(1);

        return image;
    }

    // Optimized
    public static void SetGaussianMask<T>(double sigma, ref CVImage image) where T : struct, INumber<T>
    {
        Span<T> bufferSpan = image.BufferAs<T>();

        int rX = (image.Width - 1) / 2;
        int rY = (image.Height - 1) / 2;

        double sigmaF = sigma * sigma * 2;

        int planeSize = image.Width * image.Height;

        for (int c = 0; c < image.Channels; c++)
        {
            int cBase = c * planeSize;

            double sum = 0.0;

            for (int y = -rY; y <= rY; y++)
            {
                int yOff = (y + rY) * image.Width;

                double y2 = y * y;

                for (int x = -rX; x <= rX; x++)
                {
                    double v = Math.Exp(-(x * x + y2) / sigmaF);

                    int idx = cBase + yOff + (x + rX);

                    bufferSpan[idx] = T.CreateChecked(v);
                    sum += v;
                }
            }

            T sumInv = T.CreateChecked(sum);

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

    public static CVImage CreateGaussianMask(int width = 0, int height = 0, CVDataFormat dataFormat = CVDataFormat.CV_NONE, CVChannelFormat channelFormat = CVChannelFormat.CV_None, double sigma = 1.0)
    {
        return CreateGaussianMask(width, height, dataFormat, new CVChannelFormats(channelFormat), sigma);
    }

    public static CVImage CreateGaussianMask(int width, int height, CVDataFormat dataFormat, CVChannelFormats channelFormats, double sigma = 1.0)
    {
        CVImage image = Create(width, height, dataFormat, channelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) SetGaussianMask<byte>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_S8) SetGaussianMask<sbyte>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_U16) SetGaussianMask<ushort>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_S16) SetGaussianMask<short>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_U32) SetGaussianMask<uint>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_S32) SetGaussianMask<int>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_U64) SetGaussianMask<ulong>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_S64) SetGaussianMask<long>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_F32) SetGaussianMask<float>(sigma, ref image);
        else if (image.DataFormat == CVDataFormat.CV_F64) SetGaussianMask<double>(sigma, ref image);

        return image;
    }

    public CVImage Clone()
    {
        return CreatePlanar(Width, Height, DataFormat, ChannelFormats, buffer);
    }

    public static CVImage operator +(int val, CVImage image) { return CVAdd.Add(image, val); }
    public static CVImage operator +(double val, CVImage image) { return CVAdd.Add(image, val); }
    public static CVImage operator +(CVImage image, int val) { return CVAdd.Add(image, val); }
    public static CVImage operator +(CVImage image, double val) { return CVAdd.Add(image, val); }
    public static CVImage operator +(CVImage image1, CVImage image2) { return CVAdd.Add(image1, image2); }

    // public static CVImage operator -(int val, CVImage image) { return CVSubtract.Subtract(val, image); }
    // public static CVImage operator -(double val, CVImage image) { return CVSubtract.Subtract(val, image); }
    public static CVImage operator -(CVImage image, int val) { return CVSubtract.Subtract(image, val); }
    public static CVImage operator -(CVImage image, double val) { return CVSubtract.Subtract(image, val); }
    public static CVImage operator -(CVImage image1, CVImage image2) { return CVSubtract.Subtract(image1, image2); }

    public static CVImage operator *(int val, CVImage image) { return CVMultiply.Multiply(image, val); }
    public static CVImage operator *(double val, CVImage image) { return CVMultiply.Multiply(image, val); }
    public static CVImage operator *(CVImage image, int val) { return CVMultiply.Multiply(image, val); }
    public static CVImage operator *(CVImage image, double val) { return CVMultiply.Multiply(image, val); }
    public static CVImage operator *(CVImage image1, CVImage image2) { return CVMultiply.Multiply(image1, image2); }

    // public static CVImage operator /(int val, CVImage image) { return CVDivide.Divide(val, image); }
    // public static CVImage operator /(double val, CVImage image) { return CVDivide.Divide(val, image); }
    public static CVImage operator /(CVImage image, int val) { return CVDivide.Divide(image, val); }
    public static CVImage operator /(CVImage image, double val) { return CVDivide.Divide(image, val); }
    public static CVImage operator /(CVImage image1, CVImage image2) { return CVDivide.Divide(image1, image2); }
}