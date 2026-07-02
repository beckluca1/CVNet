using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace CVNet;

public class CVCheckerboard
{
    static void DerivativeX(
        CVImage src,
        out CVImage dst)
    {
        int[,] k =
        {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };

        dst = CVImage.Create(
            src.Width,
            src.Height,
            src.ColorFormat,
            src.DataFormat,
            src.ChannelFormat);

        Span<float> srcSpan = src.BufferAs<float>();
        Span<float> dstSpan = dst.BufferAs<float>();

        for (int y = 1; y < src.Height - 1; y++)
        {
            for (int x = 1; x < src.Width - 1; x++)
            {
                float sum = 0;

                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        sum +=
                            srcSpan[(x + i) + (y + j) * src.Width] *
                            k[j + 1, i + 1];
                    }
                }
                dstSpan[x + y * dst.Width] = sum;
            }
        }
    }

    static void DerivativeY(
        CVImage src,
        out CVImage dst)
    {
        int[,] k =
        {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

        dst = CVImage.Create(
            src.Width,
            src.Height,
            src.ColorFormat,
            src.DataFormat,
            src.ChannelFormat);

        Span<float> srcSpan = src.BufferAs<float>();
        Span<float> dstSpan = dst.BufferAs<float>();

        for (int y = 1; y < src.Height - 1; y++)
        {
            for (int x = 1; x < src.Width - 1; x++)
            {
                float sum = 0;

                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        sum +=
                            srcSpan[(x + i) + (y + j) * src.Width] *
                            k[j + 1, i + 1];
                    }
                }
                dstSpan[x + y * dst.Width] = sum;
            }
        }
    }

    public static void Sobel(
        CVImage image,
        out CVImage gradientX,
        out CVImage gradientY)
    {
        int h = image.Height;
        int w = image.Width;

        gradientX = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat, image.ChannelFormat);
        gradientY = CVImage.Create(image.Width, image.Height, image.ColorFormat, image.DataFormat, image.ChannelFormat);

        int[,] sx =
        {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
        };

        int[,] sy =
        {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
        };

        Span<float> bufferSpan = image.BufferAs<float>();
        Span<float> bgx = gradientX.BufferAs<float>();
        Span<float> bgy = gradientY.BufferAs<float>();

        for (int y = 1; y < h - 1; y++)
            for (int x = 1; x < w - 1; x++)
            {
                float sumX = 0;
                float sumY = 0;

                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++)
                    {
                        float p = bufferSpan[(x + i) + (y + j) * image.Width];

                        sumX += p * sx[j + 1, i + 1];
                        sumY += p * sy[j + 1, i + 1];
                    }
                bgx[x + y * gradientX.Width] = sumX;
                bgy[x + y * gradientY.Width] = sumY;
            }
    }

    public static List<Vector<double>> DetectSaddlePoints(
    CVImage image,
    CVImage gradientX,
    CVImage gradientY,
    float saddleThreshold = 0.5f)
    {
        DerivativeX(gradientX, out CVImage ixx);
        DerivativeY(gradientY, out CVImage iyy);
        DerivativeY(gradientX, out CVImage ixy);

        int w = image.Width;
        int h = image.Height;

        int r = 3;

        var result =
            new List<Vector<double>>();

        Span<float> bxx = ixx.BufferAs<float>();
        Span<float> byy = iyy.BufferAs<float>();
        Span<float> bxy = ixy.BufferAs<float>();

        Span<float> bgx = gradientX.BufferAs<float>();
        Span<float> bgy = gradientY.BufferAs<float>();

        for (int y = r; y < h - r; y++)
        {
            for (int x = r; x < w - r; x++)
            {
                float hxx = bxx[x + y * ixx.Width];
                float hyy = byy[x + y * iyy.Width];
                float hxy = bxy[x + y * ixy.Width];

                float det =
                    hxx * hyy -
                    hxy * hxy;

                if (det > -saddleThreshold)
                    continue;

                float A = 0;
                float B = 0;
                float C = 0;

                for (int dy = -r; dy <= r; dy++)
                {
                    for (int dx = -r; dx <= r; dx++)
                    {
                        float gx = bgx[(x + dx) + (y + dy) * gradientX.Width];
                        float gy = bgy[(x + dx) + (y + dy) * gradientY.Width];

                        A += gx * gx;
                        B += gx * gy;
                        C += gy * gy;
                    }
                }

                float trace = A + C;
                float determinant =
                    A * C - B * B;

                if (determinant <= 0)
                    continue;

                float disc =
                    MathF.Sqrt(
                        Math.Max(
                            0,
                            trace * trace -
                            4 * determinant));

                float lambda1 =
                    (trace + disc) * 0.5f;

                float lambda2 =
                    (trace - disc) * 0.5f;

                if (lambda2 < lambda1 * 0.4f)
                    continue;

                result.Add(
                    DenseVector.OfArray(
                        new double[]
                        {
                        x,
                        y,
                        -det
                        }));
            }
        }

        return result;
    }

    public static List<Vector<double>> Nms(
    List<Vector<double>> points,
    float radius = 8)
    {
        List<Vector<double>> result = new List<Vector<double>>();

        foreach (var p in points.OrderByDescending(v => v[2]))
        {
            bool keep = true;

            foreach (var q in result)
            {
                double dx = p[0] - q[0];
                double dy = p[1] - q[1];

                if (dx * dx + dy * dy <
                    radius * radius)
                {
                    keep = false;
                    break;
                }
            }

            if (keep)
                result.Add(p);
        }

        return result;
    }

    public static List<Vector<double>> DetectCheckerboard(CVImage image)
    {
        CVImage gray = CVConvert.ConvertData(image, CVDataFormat.CV_F32);
        gray = CVMath.Divide(gray, 255f);
        gray = CVConvert.ToFormat(gray, CVChannelFormat.CV_Grayscale);
        gray = CVProcessing.GaussianBlur(gray, 5);

        Sobel(gray, out CVImage gx, out CVImage gy);

        List<Vector<double>> saddlePoints = DetectSaddlePoints(gray, gx, gy, 5f);
        Console.WriteLine($"Saddle points: {saddlePoints.Count}");

        if (saddlePoints.Count == 0) return new List<Vector<double>>();

        List<Vector<double>> maxPoints = Nms(saddlePoints, 4);
        Console.WriteLine($"Max points: {maxPoints.Count}");

        return maxPoints;
    }
}