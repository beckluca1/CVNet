using System.Numerics;
using System.Runtime.InteropServices;

namespace CVNet;

public class CVConvert
{
    public static void ConvertBuffer<InT, OutT>(CVImage imageIn, CVImage imageOut) where InT : struct where OutT : struct
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<InT> bufferIn = imageIn.BufferAs<InT>();
        Span<OutT> bufferOut = imageOut.BufferAs<OutT>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (OutT)Convert.ChangeType(bufferIn[i], typeof(OutT));
    }

    public static void ConvertByteToFloat(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<float> bufferOut = imageOut.BufferAs<float>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    public static void ConvertFloatToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<float> bufferIn = imageIn.BufferAs<float>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    public static void ConvertByteToInt(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<int> bufferOut = imageOut.BufferAs<int>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    public static void ConvertIntToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<int> bufferIn = imageIn.BufferAs<int>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    public static void ConvertByteToUInt(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<byte> bufferIn = imageIn.BufferAs<byte>();
        Span<uint> bufferOut = imageOut.BufferAs<uint>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = bufferIn[i];
    }

    public static void ConvertUIntToByte(CVImage imageIn, ref CVImage imageOut)
    {
        int imageDataCount = imageIn.Width * imageIn.Height * imageIn.Channels;

        Span<uint> bufferIn = imageIn.BufferAs<uint>();
        Span<byte> bufferOut = imageOut.BufferAs<byte>();

        for (int i = 0; i < imageDataCount; i++) bufferOut[i] = (byte)bufferIn[i];
    }

    public static void ConvertBuffer<InT>(CVImage imageIn, CVImage imageOut) where InT : struct
    {
        if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_F32) ConvertByteToFloat(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_F32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertFloatToByte(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_S32) ConvertByteToInt(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_S32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertIntToByte(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U8 && imageOut.DataFormat == CVDataFormat.CV_U32) ConvertByteToUInt(imageIn, ref imageOut);
        else if (imageIn.DataFormat == CVDataFormat.CV_U32 && imageOut.DataFormat == CVDataFormat.CV_U8) ConvertUIntToByte(imageIn, ref imageOut);

        else if (imageOut.DataFormat == CVDataFormat.CV_U8) ConvertBuffer<InT, byte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S8) ConvertBuffer<InT, sbyte>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U16) ConvertBuffer<InT, ushort>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S16) ConvertBuffer<InT, short>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U32) ConvertBuffer<InT, uint>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S32) ConvertBuffer<InT, int>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_U64) ConvertBuffer<InT, ulong>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_S64) ConvertBuffer<InT, long>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F32) ConvertBuffer<InT, float>(imageIn, imageOut);
        else if (imageOut.DataFormat == CVDataFormat.CV_F64) ConvertBuffer<InT, double>(imageIn, imageOut);
    }

    public static CVImage ConvertData(CVImage image, CVDataFormat dataFormat)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.ColorFormat, dataFormat, image.ChannelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) ConvertBuffer<byte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) ConvertBuffer<sbyte>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) ConvertBuffer<ushort>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) ConvertBuffer<short>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) ConvertBuffer<uint>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) ConvertBuffer<int>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) ConvertBuffer<ulong>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) ConvertBuffer<long>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) ConvertBuffer<float>(image, imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) ConvertBuffer<double>(image, imageOut);

        return imageOut;
    }

    public static CVImage ExpandFormat(CVImage image)
    {
        CVDataFormat dataFormat = CVDataFormat.CV_NONE;

        if (image.DataFormat == CVDataFormat.CV_U8) dataFormat = CVDataFormat.CV_U16;
        else if (image.DataFormat == CVDataFormat.CV_S8) dataFormat = CVDataFormat.CV_S16;
        else if (image.DataFormat == CVDataFormat.CV_U16) dataFormat = CVDataFormat.CV_U32;
        else if (image.DataFormat == CVDataFormat.CV_S16) dataFormat = CVDataFormat.CV_S32;
        else if (image.DataFormat == CVDataFormat.CV_U32) dataFormat = CVDataFormat.CV_U64;
        else if (image.DataFormat == CVDataFormat.CV_S32) dataFormat = CVDataFormat.CV_S64;
        else if (image.DataFormat == CVDataFormat.CV_U64) dataFormat = CVDataFormat.CV_U64;
        else if (image.DataFormat == CVDataFormat.CV_S64) dataFormat = CVDataFormat.CV_S64;
        else if (image.DataFormat == CVDataFormat.CV_F32) dataFormat = CVDataFormat.CV_F64;
        else if (image.DataFormat == CVDataFormat.CV_F64) dataFormat = CVDataFormat.CV_F64;

        return ConvertData(image, dataFormat);
    }

    private static void CopyChannel(CVImage imageIn, CVImage imageOut, int channelIn, int channelOut)
    {
        int byteCount = imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetIn = channelIn * imageIn.Width * imageIn.Height * imageIn.Bytes;
        int offsetOut = channelOut * imageOut.Width * imageOut.Height * imageOut.Bytes;

        Buffer.BlockCopy(imageIn.buffer, offsetIn, imageOut.buffer, offsetOut, byteCount);
    }

    private static void FillChannel<OutT, T>(CVImage imageIn, int channel, T value) where OutT : struct where T : struct
    {
        Span<OutT> bufferSpan = imageIn.BufferAs<OutT>();

        OutT valueConv = (OutT)Convert.ChangeType(value, typeof(OutT));

        bufferSpan.Slice(channel * imageIn.Width * imageIn.Height, imageIn.Width * imageIn.Height).Fill(valueConv);
    }

    private static void FillChannel<T>(CVImage image, int channel, T value) where T : struct
    {
        if (image.DataFormat == CVDataFormat.CV_U8) FillChannel<byte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S8) FillChannel<sbyte, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U16) FillChannel<ushort, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S16) FillChannel<short, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U32) FillChannel<uint, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S32) FillChannel<int, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_U64) FillChannel<ulong, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_S64) FillChannel<long, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F32) FillChannel<float, T>(image, channel, value);
        else if (image.DataFormat == CVDataFormat.CV_F64) FillChannel<double, T>(image, channel, value);
    }

    public static CVImage ConvertColor(CVImage image, int[] inChannels)
    {
        CVColorFormat format = (CVColorFormat)inChannels.Length;

        CVImage outImage = CVImage.Create(image.Width, image.Height, format, image.DataFormat, image.ChannelFormat);

        for (int i = 0; i < inChannels.Length; i++)
        {
            CopyChannel(image, outImage, inChannels[i], i);
        }

        return outImage;
    }

    public static CVImage AverageChannels(CVImage image, CVChannelFormat channelFormat)
    {
        // Expand so sum doesnt fail
        CVImage expanded = ExpandFormat(image);

        CVChannelFormats channelFormats = new CVChannelFormats(channelFormat);
        Dictionary<CVChannel, CVImage> uniqueChannels = new Dictionary<CVChannel, CVImage>();
        for (int i = 0; i < channelFormats.Channels.Length; i++)
        {
            if (uniqueChannels.ContainsKey(channelFormats.Channels[i])) continue;

            for (int j = 0; j < expanded.ChannelFormats.Channels.Length; j++)
            {
                if (channelFormats.Channels[i] == expanded.ChannelFormats.Channels[j])
                {
                    uniqueChannels.Add(channelFormats.Channels[i], ConvertColor(expanded, [j]));
                    break;
                }
            }
        }

        if (uniqueChannels.Count == 0) throw new Exception("No RGB components found");

        CVImage sum = uniqueChannels.Values.ElementAt(0);
        sum.ChannelFormat = CVChannelFormat.CV_R;
        for (int i = 1; i < uniqueChannels.Count; i++)
        {
            sum = CVAdd.Add(sum, uniqueChannels.Values.ElementAt(i));
        }
        sum = CVDivide.Divide(sum, uniqueChannels.Count);

        //Recast to original type
        sum = ConvertData(sum, image.DataFormat);

        return sum;
    }

    public static void ToFormat<T>(CVImage image, ref CVImage imageOut) where T : struct, INumber<T>
    {
        int[] requestedChannels = new int[imageOut.ChannelFormats.Channels.Length];
        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
            for (int j = 0; j < image.ChannelFormats.Channels.Length; j++)
            {
                if (imageOut.ChannelFormats.Channels[i] == image.ChannelFormats.Channels[j])
                {
                    requestedChannels[i] = j;
                    break;
                }
            }

        imageOut = ConvertColor(image, requestedChannels);

        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
        {
            if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_AVG_RGB)
            {
                CVImage average = AverageChannels(image, CVChannelFormat.CV_RGB);

                CopyChannel(average, imageOut, 0, i);
            }
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_AVG_RGBA)
            {
                CVImage average = AverageChannels(image, CVChannelFormat.CV_RGBA);
                CopyChannel(average, imageOut, 0, i);
            }
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_ZERO)
                FillChannel(imageOut, i, 0);
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_ONE)
                FillChannel(imageOut, i, 1);
            else if (imageOut.ChannelFormats.Channels[i] == CVChannel.CV_A_255)
                FillChannel(imageOut, i, 255);
        }

        // Replace Placeholder Channel Types
        for (int i = 0; i < imageOut.ChannelFormats.Channels.Length; i++)
        {
            if (imageOut.ChannelFormats.Channels[i] >= CVChannel.CV_AVG_RGB && imageOut.ChannelFormats.Channels[i] <= CVChannel.CV_AVG_RGBA)
                imageOut.ChannelFormats.Channels[i] = CVChannel.CV_R;
            else if (imageOut.ChannelFormats.Channels[i] >= CVChannel.CV_A_ZERO && imageOut.ChannelFormats.Channels[i] <= CVChannel.CV_A_255)
                imageOut.ChannelFormats.Channels[i] = CVChannel.CV_A;
        }
    }

    public static CVImage ToFormat(CVImage image, CVChannelFormat channelFormat)
    {
        CVImage imageOut = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat, channelFormat);

        if (image.DataFormat == CVDataFormat.CV_U8) ToFormat<byte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S8) ToFormat<sbyte>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U16) ToFormat<ushort>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S16) ToFormat<short>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U32) ToFormat<uint>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S32) ToFormat<int>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_U64) ToFormat<ulong>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_S64) ToFormat<long>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F32) ToFormat<float>(image, ref imageOut);
        else if (image.DataFormat == CVDataFormat.CV_F64) ToFormat<double>(image, ref imageOut);

        return imageOut;
    }
}