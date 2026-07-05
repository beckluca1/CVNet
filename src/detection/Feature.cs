using System.Numerics;

namespace CVNet;

class CVFeatureDetector
{
    public static uint Compare<T>(T centerVal, T otherVal, T threshold) where T : INumber<T>
    {
        if (otherVal >= centerVal)
            return otherVal - centerVal > threshold ? 2u : 1u;
        else
            return centerVal - otherVal > threshold ? 0u : 1u;
    }

    public static bool CheckConsecutive(uint bitMask)
    {
        for (int i = 1; i < 12; i++)
            bitMask &= (bitMask >> i);

        return bitMask != 0;
    }

    public static void Fast<T>(CVImage image, T threshold) where T : struct, INumber<T>
    {
        int[] xOffsets = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
        int[] yOffsets = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3,];

        int[] earlyChecks = [1, 5, 9, 13];
        int[] lateChecks = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15];

        Span<T> buffer = image.BufferAs<T>();

        List<(int, int)> keypointIndics = new List<(int, int)>();

        for (int x = 3; x < image.Width - 3; x++)
            for (int y = 3; y < image.Width - 3; y++)
            {
                T centerValue = buffer[x + image.Width * y];

                // Bit masks
                uint[] bitMasks = new uint[3];

                uint[] pixelVals = new uint[3];
                for (int r = 0; r < earlyChecks.Length; r++)
                {
                    uint state = Compare<T>(centerValue, buffer[(x + xOffsets[earlyChecks[r]]) + image.Width * (y + yOffsets[earlyChecks[r]])], threshold);
                    pixelVals[1 + state]++;
                    bitMasks[1 + state] |= (1u << earlyChecks[r]);
                }

                // Early rejection
                if (pixelVals[0] < 3 && pixelVals[2] < 3) continue;

                for (int r = 0; r < lateChecks.Length; r++)
                {
                    uint state = Compare<T>(centerValue, buffer[(x + xOffsets[lateChecks[r]]) + image.Width * (y + yOffsets[lateChecks[r]])], threshold);
                    bitMasks[1 + state] |= (1u << lateChecks[r]);
                }

                uint darkMask = bitMasks[0] | (bitMasks[0] << 16);
                uint brightMask = bitMasks[2] | (bitMasks[2] << 16);

                if (CheckConsecutive(darkMask) || CheckConsecutive(brightMask))
                {
                    keypointIndics.Add((x, y));
                }
            }
    }

    public static void HarrisStrength(CVImage image)
    {

    }

    public static void Orb(CVImage image)
    {

    }
}