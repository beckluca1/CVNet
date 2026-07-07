#if DEBUG
namespace CVNet;

using Xunit;

public class ArucoTests
{
    [Fact]
    public void TestGetMarker()
    {
        // Create white image and fill with marker like pixels
        CVImage image = CVImage.Create<uint>(256, 256, CVDataFormat.CV_U32, CVChannelFormat.CV_R, 120);

        Span<uint> imageBuffer = image.BufferAs<uint>();

        int size = 10;
        int markerSize = 8;
        int boundary = size;
        int xOffset = boundary + 1;
        int yOffset = boundary + 1;
        int xSize = size;
        int ySize = size;

        byte[] markerPattern = [120, 120, 120, 120, 120, 120, 120, 120,
                                120, 120, 255, 255, 120, 120, 120, 120,
                                120, 120, 255, 120, 255, 120, 255, 120,
                                120, 255, 120, 120, 255, 120, 120, 120,
                                120, 120, 120, 120, 255, 255, 120, 120,
                                120, 255, 120, 120, 255, 255, 120, 120,
                                120, 255, 255, 255, 120, 255, 120, 120,
                                120, 120, 120, 120, 120, 120, 120, 120 ];

        for (int i = 0; i < markerSize * xSize + 2 * boundary; i++)
            for (int j = 0; j < markerSize * ySize + 2 * boundary; j++)
                imageBuffer[(xOffset - boundary + i) + (yOffset - boundary + j) * image.Width] = 255;

        for (int i = 0; i < markerSize; i++)
            for (int j = 0; j < markerSize; j++)
                for (int y = 0; y < ySize; y++)
                    for (int x = 0; x < xSize; x++)
                        imageBuffer[(xOffset + i * xSize + x) + (yOffset + j * ySize + y) * image.Width] = markerPattern[j * markerSize + i];

        CVAruco.GetMarker(image, 6);
    }
}
#endif