using System.Numerics;
using Xunit.Sdk;

namespace CVNet;

public class CVDescriptor
{
    public (int x, int y) Position;
    public ulong[] Descriptor;
    public int Bits;
    public int Words;

    public CVDescriptor((int x, int y) position, int bits = 256)
    {
        Bits = bits;
        Words = (int)Math.Ceiling(Bits / 64.0);
        Position = position;
        Descriptor = new ulong[Words];
    }

    public CVDescriptor((int x, int y) position, ulong[] descriptor)
    {
        Words = descriptor.Length;
        Bits = Words * 64;
        Position = position;
        Descriptor = descriptor;
    }

    public bool this[int index]
    {
        get => (Descriptor[index / 64] & (1UL << (index % 64))) != 0;
        set
        {
            int word = index / 64;
            ulong mask = 1UL << (index % 64);

            if (value)
                Descriptor[word] |= mask;
            else
                Descriptor[word] &= ~mask;
        }
    }

    public int Distance(CVDescriptor other)
    {
        int distance = 0;

        // Count valid words
        for (int i = 0; i < Words; i++)
        {
            distance += BitOperations.PopCount(Descriptor[i] ^ other.Descriptor[i]);
        }

        // Remove padding bits from the last word
        int unusedBits = (Words * 64) - Bits;

        if (unusedBits > 0)
        {
            ulong mask = ulong.MaxValue >> unusedBits;
            ulong paddedDifference = Descriptor[Words - 1] ^ other.Descriptor[other.Words - 1];
            int oldContribution = BitOperations.PopCount(paddedDifference);
            int newContribution = BitOperations.PopCount(paddedDifference & mask);

            distance -= oldContribution;
            distance += newContribution;
        }

        return distance;
    }
}

class KDNode
{
    public readonly CVDescriptor? Descriptor = null;
    public readonly int Axis;
    public readonly KDNode? Left = null;
    public readonly KDNode? Right = null;

    int Partition(List<int> indices, int left, int right, List<CVDescriptor> descriptors, int axis)
    {
        int pivotIndex = right;
        int pivotValue = axis == 0
            ? descriptors[indices[pivotIndex]].Position.x
            : descriptors[indices[pivotIndex]].Position.y;

        int store = left;

        for (int i = left; i < right; i++)
        {
            int value = axis == 0
                ? descriptors[indices[i]].Position.x
                : descriptors[indices[i]].Position.y;

            if (value < pivotValue)
            {
                (indices[store], indices[i]) = (indices[i], indices[store]);
                store++;
            }
        }

        (indices[store], indices[right]) = (indices[right], indices[store]);

        return store;
    }

    void QuickSelect(List<int> indices,
                     int left,
                     int right,
                     int k,
                     List<CVDescriptor> descriptors,
                     int axis)
    {
        while (left < right)
        {
            int pivot = Partition(indices, left, right, descriptors, axis);

            if (pivot == k)
                return;

            if (k < pivot)
                right = pivot - 1;
            else
                left = pivot + 1;
        }
    }

    public KDNode(List<CVDescriptor> descriptors) : this(descriptors, Enumerable.Range(0, descriptors.Count).ToList(), 0, descriptors.Count - 1, 0) { }

    private KDNode(List<CVDescriptor> descriptors, List<int> indices, int start, int end, int depth)
    {
        if (start > end)
            return;

        int axis = depth % 2;
        int mid = (start + end) / 2;
        QuickSelect(indices, start, end, mid, descriptors, axis);

        Descriptor = descriptors[indices[mid]];
        Axis = axis;
        if (mid > start)
            Left = new KDNode(descriptors, indices, start, mid - 1, depth + 1);
        if (mid < end)
            Right = new KDNode(descriptors, indices, mid + 1, end, depth + 1);
    }

    public void RadiusSearch(int qx, int qy, int radiusSquared, ref List<CVDescriptor> results)
    {
        if (Descriptor == null)
            return;

        int dx = Descriptor.Position.x - qx;
        int dy = Descriptor.Position.y - qy;

        int distSquared = dx * dx + dy * dy;

        if (distSquared <= radiusSquared)
            results.Add(Descriptor);

        int diff;
        if (Axis == 0)
            diff = qx - Descriptor.Position.x;
        else
            diff = qy - Descriptor.Position.y;

        KDNode? near;
        KDNode? far;

        if (diff <= 0)
        {
            near = Left;
            far = Right;
        }
        else
        {
            near = Right;
            far = Left;
        }

        if (near != null)
            near.RadiusSearch(qx, qy, radiusSquared, ref results);

        if (far != null && diff * diff <= radiusSquared)
        {
            far.RadiusSearch(qx, qy, radiusSquared, ref results);
        }
    }
}

class BinaryLSH
{
    private class HashTable
    {
        public Dictionary<int, List<int>> Buckets = new();
        public int[] Bits;

        public HashTable(int bits)
        {
            Bits = new int[bits];
        }
    }

    private readonly List<CVDescriptor> descriptors;
    private readonly List<HashTable> tables = new();

    private readonly Random random = new();

    private const int TABLES = 8;
    private const int HASH_BITS = 16;

    public BinaryLSH(List<CVDescriptor> descriptors)
    {
        if (descriptors == null || descriptors.Count == 0)
            throw new ArgumentException("Descriptor list is empty");

        this.descriptors = descriptors;

        for (int t = 0; t < TABLES; t++)
        {
            var table = new HashTable(HASH_BITS);

            for (int i = 0; i < HASH_BITS; i++)
            {
                // Select random bit positions only inside valid descriptor bits
                table.Bits[i] = random.Next(descriptors[0].Bits);
            }

            tables.Add(table);
        }

        Build();
    }

    private void Build()
    {
        for (int i = 0; i < descriptors.Count; i++)
        {
            foreach (var table in tables)
            {
                int key = Hash(descriptors[i], table.Bits);

                if (!table.Buckets.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    table.Buckets[key] = list;
                }

                list.Add(i);
            }
        }
    }

    private int Hash(CVDescriptor d, int[] bits)
    {
        int hash = 0;

        for (int i = 0; i < bits.Length; i++)
        {
            hash |= (d[bits[i]] ? 1 : 0) << i;
        }

        return hash;
    }


    private HashSet<int> CollectCandidates(CVDescriptor query)
    {
        HashSet<int> candidates = new();

        foreach (var table in tables)
        {
            int key = Hash(query, table.Bits);

            if (table.Buckets.TryGetValue(key, out var bucket))
            {
                foreach (int id in bucket)
                    candidates.Add(id);
            }
        }

        return candidates;
    }

    public void FindBestBound(
        CVDescriptor query,
        int maxSquaredDistance,
        out int bestIndex,
        out int bestDistance,
        out int secondDistance)
    {
        HashSet<int> candidates = CollectCandidates(query);

        bestIndex = -1;
        bestDistance = int.MaxValue;
        secondDistance = int.MaxValue;

        for (int i = 0; i < descriptors.Count; i++)
        {
            int dx = descriptors[i].Position.x - query.Position.x;
            int dy = descriptors[i].Position.y - query.Position.y;

            if (dx * dx + dy * dy > maxSquaredDistance)
                continue;

            int distance = query.Distance(descriptors[i]);
            UpdateBest(i, distance, ref bestIndex, ref bestDistance, ref secondDistance);
        }
    }


    public void FindBestUnbound(
        CVDescriptor query,
        out int bestIndex,
        out int bestDistance,
        out int secondDistance)
    {
        HashSet<int> candidates = CollectCandidates(query);

        bestIndex = -1;
        bestDistance = int.MaxValue;
        secondDistance = int.MaxValue;

        for (int i = 0; i < descriptors.Count; i++)
        {
            int distance = query.Distance(descriptors[i]);
            UpdateBest(i, distance, ref bestIndex, ref bestDistance, ref secondDistance);
        }
    }


    private void UpdateBest(
        int id,
        int distance,
        ref int bestIndex,
        ref int bestDistance,
        ref int secondDistance)
    {
        if (distance < bestDistance)
        {
            secondDistance = bestDistance;
            bestDistance = distance;
            bestIndex = id;
        }
        else if (distance < secondDistance)
        {
            secondDistance = distance;
        }
    }
}

public class CVFeatureDetector
{
    public static List<(CVImage image, double scale)> BuildPyramid(CVImage image, int levels = 8, double scaleFactor = 0.8)
    {
        List<(CVImage, double)> pyramid = new List<(CVImage, double)>();

        double scale = 1.0;

        pyramid.Add((image, scale));

        for (int i = 1; i < levels; i++)
        {
            scale *= scaleFactor;

            int newWidth = (int)(image.Width * scale);
            int newHeight = (int)(image.Height * scale);

            if (newWidth < 32 || newHeight < 32)
                break;

            CVImage nextImage = CVBlur.GaussianBlur(image, 2);
            nextImage = CVResize.StretchLinear(nextImage, newWidth, newHeight);
            pyramid.Add((nextImage, scale));
        }

        return pyramid;
    }

    private static uint fastCompare<T>(T centerVal, T otherVal, T threshold) where T : INumber<T>
    {
        if (otherVal >= centerVal)
            return otherVal - centerVal > threshold ? 2u : 1u;
        else
            return centerVal - otherVal > threshold ? 0u : 1u;
    }

    private static bool fastCheckConsecutive(uint bitMask)
    {
        for (int i = 0; i < 16; i++)
        {
            uint test = (bitMask >> i) & 0x1FF; // 9 bits
            if (test == 0x1FF)
                return true;
        }

        return false;
    }
    private static int[] xOffsets = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    private static int[] yOffsets = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3,];

    private static int[] earlyChecks = [1, 5, 9, 13];
    private static int[] lateChecks = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15];

    private static void fast<T, TV>(CVImage image, TV threshold, int edgeDistance, ref List<(int, int)> keypoints) where T : struct, INumber<T> where TV : struct, INumber<TV>
    {
        T thresholdT = T.CreateChecked(threshold);

        Span<T> buffer = image.BufferAs<T>();

        for (int x = edgeDistance; x < image.Width - edgeDistance; x++)
            for (int y = edgeDistance; y < image.Height - edgeDistance; y++)
            {
                T centerValue = buffer[x + image.Width * y];

                // Bit masks
                uint darkMask = 0;
                uint brightMask = 0;

                // Counter
                int darkCount = 0;
                int brightCount = 0;

                for (int r = 0; r < earlyChecks.Length; r++)
                {
                    uint state = fastCompare<T>(centerValue, buffer[(x + xOffsets[earlyChecks[r]]) + image.Width * (y + yOffsets[earlyChecks[r]])], thresholdT);

                    if (state == 0)
                    {
                        darkCount++;
                        darkMask |= (1u << earlyChecks[r]);
                    }
                    else if (state == 2)
                    {
                        brightCount++;
                        brightMask |= (1u << earlyChecks[r]);
                    }
                }

                // Early rejection
                if (darkCount < 3 && brightCount < 3) continue;

                for (int r = 0; r < lateChecks.Length; r++)
                {
                    uint state = fastCompare<T>(centerValue, buffer[(x + xOffsets[lateChecks[r]]) + image.Width * (y + yOffsets[lateChecks[r]])], thresholdT);

                    if (state == 0)
                    {
                        darkCount++;
                        darkMask |= (1u << lateChecks[r]);
                    }
                    else if (state == 2)
                    {
                        brightCount++;
                        brightMask |= (1u << lateChecks[r]);
                    }
                }

                // Late-Early rejection
                if (darkCount < 9 && brightCount < 9) continue;

                darkMask = darkMask | (darkMask << 16);
                brightMask = brightMask | (brightMask << 16);

                if (fastCheckConsecutive(darkMask) || fastCheckConsecutive(brightMask))
                {
                    keypoints.Add((x, y));
                }
            }
    }

    public static List<(int, int)> Fast<T>(CVImage image, T threshold, int edgeDistance = 16) where T : struct, INumber<T>
    {
        List<(int, int)> keypoints = new List<(int, int)>();

        if (image.DataFormat == CVDataFormat.CV_U8) fast<byte, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S8) fast<sbyte, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U16) fast<ushort, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S16) fast<short, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U32) fast<uint, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S32) fast<int, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_U64) fast<ulong, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_S64) fast<long, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_F32) fast<float, T>(image, threshold, edgeDistance, ref keypoints);
        else if (image.DataFormat == CVDataFormat.CV_F64) fast<double, T>(image, threshold, edgeDistance, ref keypoints);

        return keypoints;
    }

    public static List<double> Angles(List<(int x, int y)> points, List<(double x, double y)> centroids)
    {
        List<double> angles = new List<double>();

        for (int i = 0; i < points.Count; i++)
        {
            angles.Add(Math.Atan2(centroids[i].y - points[i].y, centroids[i].x - points[i].x));
        }

        return angles;
    }

    private static double intensityCentroidAngleSingle<T>(CVImage image, (int x, int y) point, int radius) where T : struct, INumber<T>
    {
        double m10 = 0.0;
        double m01 = 0.0;

        Span<T> buffer = image.BufferAs<T>();

        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                if (x * x + y * y > radius * radius)
                    continue;

                int px = point.x + x;
                int py = point.y + y;

                if (px < 0 || px >= image.Width ||
                    py < 0 || py >= image.Height)
                    continue;

                double intensity = double.CreateChecked(buffer[px + image.Width * py]);

                // ORB moments
                m10 += x * intensity;
                m01 += y * intensity;
            }
        }

        return Math.Atan2(m01, m10);
    }

    private static void intensityCentroidAngles<T>(CVImage image, List<(int, int)> points, int radius, ref List<double> angles) where T : struct, INumber<T>
    {
        for (int i = 0; i < points.Count; i++)
        {
            angles.Add(intensityCentroidAngleSingle<T>(image, points[i], radius));
        }
    }

    public static List<double> IntensityCentroidAngles(CVImage image, List<(int, int)> points, int radius)
    {
        List<double> angles = new List<double>();

        if (image.DataFormat == CVDataFormat.CV_U8) intensityCentroidAngles<byte>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_S8) intensityCentroidAngles<sbyte>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_U16) intensityCentroidAngles<ushort>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_S16) intensityCentroidAngles<short>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_U32) intensityCentroidAngles<uint>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_S32) intensityCentroidAngles<int>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_U64) intensityCentroidAngles<ulong>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_S64) intensityCentroidAngles<long>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_F32) intensityCentroidAngles<float>(image, points, radius, ref angles);
        else if (image.DataFormat == CVDataFormat.CV_F64) intensityCentroidAngles<double>(image, points, radius, ref angles);

        return angles;
    }

    public static int[] bit_pattern_31 =
    {
        8,-3, 9,5/*mean (0), correlation (0)*/,
        4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
        -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
        7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
        2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
        1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
        -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
        -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
        -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
        10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
        -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
        -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
        7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
        -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
        -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
        -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
        12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
        -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
        -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
        11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
        4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
        5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
        3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
        -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
        -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
        -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
        -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
        -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
        -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
        5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
        5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
        1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
        9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
        4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
        2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
        -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
        -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
        4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
        0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
        -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
        -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
        -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
        8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
        0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
        7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
        -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
        10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
        -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
        10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
        -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
        -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
        3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
        5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
        -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
        3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
        2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
        -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
        -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
        -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
        -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
        6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
        -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
        -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
        -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
        3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
        -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
        -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
        2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
        -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
        -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
        5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
        -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
        -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
        -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
        10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
        7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
        -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
        -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
        7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
        -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
        -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
        -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
        7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
        -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
        1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
        2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
        -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
        -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
        7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
        1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
        9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
        -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
        -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
        7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
        12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
        6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
        5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
        2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
        3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
        2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
        9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
        -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
        -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
        1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
        6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
        2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
        6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
        3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
        7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
        -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
        -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
        -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
        -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
        8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
        4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
        -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
        4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
        -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
        -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
        7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
        -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
        -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
        8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
        -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
        1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
        7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
        -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
        11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
        -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
        3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
        5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
        0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
        -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
        0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
        -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
        5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
        3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
        -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
        -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
        -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
        6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
        -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
        -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
        1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
        4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
        -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
        2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
        -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
        4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
        -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
        -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
        7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
        4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
        -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
        7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
        7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
        -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
        -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
        -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
        2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
        10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
        -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
        8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
        2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
        -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
        -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
        -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
        5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
        -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
        -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
        -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
        -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
        -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
        2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
        -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
        -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
        -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
        -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
        6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
        -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
        11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
        7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
        -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
        -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
        -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
        -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
        -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
        -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
        -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
        -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
        1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
        1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
        9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
        5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
        -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
        -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
        -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
        -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
        8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
        2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
        7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
        -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
        -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
        4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
        3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
        -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
        5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
        4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
        -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
        0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
        -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
        3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
        -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
        8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
        -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
        2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
        10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
        6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
        -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
        -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
        -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
        -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
        -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
        4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
        2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
        6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
        3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
        11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
        -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
        4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
        2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
        -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
        -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
        -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
        6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
        0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
        -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
        -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
        -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
        5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
        2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
        -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
        9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
        11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
        3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
        -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
        3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
        -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
        5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
        8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
        7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
        -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
        7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
        9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
        7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
        -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
    };

    private static double briefSample<T>(CVImage image, double x, double y) where T : struct, INumber<T>
    {
        Span<T> buffer = image.BufferAs<T>();

        if (x < 0 || y < 0 || x >= image.Width - 2 || y >= image.Height - 2)
            return 0;

        int x0 = (int)(x);
        int y0 = (int)(y);

        double dx = x - x0;
        double dy = y - y0;

        double i00 = double.CreateChecked(buffer[(x0 + 0) + (y0 + 0) * image.Width]);
        double i10 = double.CreateChecked(buffer[(x0 + 1) + (y0 + 0) * image.Width]);
        double i01 = double.CreateChecked(buffer[(x0 + 0) + (y0 + 1) * image.Width]);
        double i11 = double.CreateChecked(buffer[(x0 + 1) + (y0 + 1) * image.Width]);

        return i00 * (1 - dx) * (1 - dy) + i10 * dx * (1 - dy) + i01 * (1 - dx) * dy + i11 * dx * dy;
    }

    private static CVDescriptor briefSingle<T>(CVImage image, (int x, int y) point, double angle) where T : struct, INumber<T>
    {
        double cosA = Math.Cos(angle);
        double sinA = Math.Sin(angle);

        CVDescriptor briefDescriptor = new CVDescriptor(point);
        for (int i = 0; i < 256; i++)
        {
            int offset = i * 4;

            int index0X = bit_pattern_31[offset + 0];
            int index0Y = bit_pattern_31[offset + 1];
            int index1X = bit_pattern_31[offset + 2];
            int index1Y = bit_pattern_31[offset + 3];

            double rotated0X = point.x + (index0X * cosA - index0Y * sinA);
            double rotated0Y = point.y + (index0X * sinA + index0Y * cosA);
            double rotated1X = point.x + (index1X * cosA - index1Y * sinA);
            double rotated1Y = point.y + (index1X * sinA + index1Y * cosA);

            double imageP1 = briefSample<T>(image, rotated0X, rotated0Y);
            double imageP2 = briefSample<T>(image, rotated1X, rotated1Y);
            briefDescriptor[i] = imageP1 <= imageP2;
        }

        return briefDescriptor;
    }

    private static void brief<T>(CVImage image, List<(int x, int y)> points, List<double> angles, ref List<CVDescriptor> descriptors) where T : struct, INumber<T>
    {
        for (int i = 0; i < points.Count; i++)
        {
            descriptors.Add(briefSingle<T>(image, points[i], angles[i]));
        }
    }

    public static List<CVDescriptor> Brief(CVImage image, List<(int, int)> points, List<double> angles)
    {
        List<CVDescriptor> descriptors = new List<CVDescriptor>();

        if (image.DataFormat == CVDataFormat.CV_U8) brief<byte>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S8) brief<sbyte>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U16) brief<ushort>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S16) brief<short>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U32) brief<uint>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S32) brief<int>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U64) brief<ulong>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S64) brief<long>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F32) brief<float>(image, points, angles, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F64) brief<double>(image, points, angles, ref descriptors);

        return descriptors;
    }

    public static List<(int x, int y)> GridKeypointSelection(
        CVImage image,
        List<(int x, int y, double score)> harrisScores,
        int gridCols = 8,
        int gridRows = 8,
        int maxPerCell = 10)
    {
        List<(int x, int y)> selected = new();

        // Create grid cells
        List<(int x, int y, double score)>[,] grid = new List<(int x, int y, double score)>[gridCols, gridRows];

        for (int x = 0; x < gridCols; x++)
        {
            for (int y = 0; y < gridRows; y++)
            {
                grid[x, y] = new List<(int x, int y, double score)>();
            }
        }

        // Assign keypoints to cells
        for (int i = 0; i < harrisScores.Count; i++)
        {
            int cellX = Math.Clamp(
                harrisScores[i].x * gridCols / image.Width,
                0,
                gridCols - 1);

            int cellY = Math.Clamp(
                harrisScores[i].y * gridRows / image.Height,
                0,
                gridRows - 1);

            grid[cellX, cellY].Add((harrisScores[i].x, harrisScores[i].y, harrisScores[i].score));
        }

        // Keep strongest points in each cell
        for (int x = 0; x < gridCols; x++)
        {
            for (int y = 0; y < gridRows; y++)
            {
                var cell = grid[x, y];

                if (cell.Count == 0)
                    continue;

                cell.Sort((a, b) => b.score.CompareTo(a.score));

                int count = Math.Min(maxPerCell, cell.Count);

                for (int i = 0; i < count; i++)
                {
                    selected.Add((cell[i].x, cell[i].y));
                }
            }
        }

        return selected;
    }

    public static List<CVDescriptor> Orb(CVImage image)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);

        List<(CVImage image, double scale)> pyramid = BuildPyramid(gray);

        List<CVDescriptor> descriptors = new();

        for (int level = 0; level < pyramid.Count; level++)
        {
            CVImage levelImage = pyramid[level].image;
            double scale = pyramid[level].scale;

            int levelFeatures = (int)(1000 * Math.Pow(scale, 0.3));

            int desiredPerCell = 3;

            int cells = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(levelFeatures / (double)desiredPerCell)));

            int cellsX = Math.Max(1, (int)Math.Round(cells * levelImage.Width / (double)levelImage.Height));
            int cellsY = Math.Max(1, cells);

            int perCell = Math.Max(1, (int)Math.Ceiling(levelFeatures / (double)(cellsX * cellsY)));

            List<(int x, int y)> fastKeypoints = Fast(levelImage, 8.0, 8);
            List<(int x, int y, double)> harrisScores = CVCornerDetector.HarrisStrengthPoints(levelImage, fastKeypoints, 0.04, 3);
            List<(int x, int y, double)> suppressedScores = CVCornerDetector.NonMaximumSuppressionPoints(harrisScores, 1);
            List<(int x, int y)> selectedPoints = GridKeypointSelection(levelImage, harrisScores, cellsX, cellsY, perCell);

            List<double> angles = IntensityCentroidAngles(levelImage, selectedPoints, 15);
            List<CVDescriptor> levelDescriptors = Brief(levelImage, selectedPoints, angles);

            for (int i = 0; i < levelDescriptors.Count; i++)
            {
                var d = levelDescriptors[i];

                // Convert back to original coordinates
                int x = (int)(selectedPoints[i].x / scale);
                int y = (int)(selectedPoints[i].y / scale);

                d.Position = (x, y);

                descriptors.Add(d);
            }
        }

        return descriptors;
    }

    public static void MatchFeaturesOrb(
            CVImage image1,
            CVImage image2,
            int hammingDistance,
            out List<(int x, int y)> matchedFeatures1,
            out List<(int x, int y)> matchedFeatures2)
    {
        List<CVDescriptor> features1 = Orb(image1);
        List<CVDescriptor> features2 = Orb(image2);

        matchedFeatures1 = new List<(int x, int y)>();
        matchedFeatures2 = new List<(int x, int y)>();

        if (features1.Count == 0 || features2.Count == 0)
            return;

        int[] best12 = new int[features1.Count];
        int[] best21 = new int[features2.Count];

        int[] dist12 = new int[features1.Count];
        int[] dist21 = new int[features2.Count];

        // Build LSH indexes
        BinaryLSH lsh2 = new BinaryLSH(features2);
        BinaryLSH lsh1 = new BinaryLSH(features1);

        // features1 -> features2
        for (int i = 0; i < features1.Count; i++)
            lsh2.FindBestUnbound(features1[i], out best12[i], out dist12[i], out _);

        // features2 -> features1
        for (int i = 0; i < features2.Count; i++)
            lsh1.FindBestUnbound(features2[i], out best21[i], out dist21[i], out _);

        // Cross-check + threshold
        for (int i = 0; i < features1.Count; i++)
        {
            int j = best12[i];

            if (j < 0)
                continue;

            // Mutual nearest neighbor
            if (i != best21[j])
                continue;

            int distance = dist12[i];
            if (distance > hammingDistance)
                continue;

            matchedFeatures1.Add(features1[i].Position);
            matchedFeatures2.Add(features2[j].Position);
        }
    }

    private static void census<T>(CVImage image, int radius, ref CVImage outImage) where T : struct, INumber<T>
    {
        Span<T> buffer = image.BufferAs<T>();
        Span<ulong> outBuffer = outImage.BufferAs<ulong>();

        int planeSize = image.Width * image.Height;

        for (int c = 0; c < image.Channels; c++)
        {
            int channelOffset = c * planeSize;
            for (int y = radius; y < image.Height - radius; y++)
            {
                int row = channelOffset + y * image.Width;
                for (int x = radius; x < image.Width - radius; x++)
                {
                    int size = 2 * radius + 1;
                    CVDescriptor descriptor = new CVDescriptor((x, y), size * size - 1);
                    int center = row + x;
                    T centerVal = buffer[center];
                    int index = 0;
                    for (int dy = -radius; dy <= radius; dy++)
                    {
                        int srcRow = channelOffset + (y + dy) * image.Width;
                        for (int dx = -radius; dx <= radius; dx++)
                        {
                            if (dy == 0 && dx == 0) continue;
                            T pixelVal = buffer[srcRow + x + dx];
                            descriptor[index] = pixelVal >= centerVal;
                            index++;
                        }
                    }

                    // Only works for images <= 8x8 so radius <= 3
                    outBuffer[center] = descriptor.Descriptor[0];
                }
            }
        }
    }

    public static CVImage Census(CVImage image, int radius)
    {
        CVImage descriptorImage = CVImage.Create(image.Width, image.Height, CVDataFormat.CV_U64, image.ChannelFormats);

        if (image.DataFormat == CVDataFormat.CV_U8) census<byte>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_S8) census<sbyte>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_U16) census<ushort>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_S16) census<short>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_U32) census<uint>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_S32) census<int>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_U64) census<ulong>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_S64) census<long>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_F32) census<float>(image, radius, ref descriptorImage);
        else if (image.DataFormat == CVDataFormat.CV_F64) census<double>(image, radius, ref descriptorImage);

        return descriptorImage;
    }

    private static void censusFeatures<T>(CVImage image, int radius, int offsetX, int y, ref List<CVDescriptor> descriptors) where T : struct, INumber<T>
    {
        Span<T> buffer = image.BufferAs<T>();
        y = Math.Min(Math.Max(y, radius), image.Height - 1 - radius);
        int row = y * image.Width;
        for (int x = radius; x < image.Width - radius; x += offsetX)
        {
            int size = 2 * radius + 1;
            CVDescriptor descriptor = new CVDescriptor((x, y), size * size - 1);
            int center = row + x;
            T centerVal = buffer[center];
            int index = 0;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int srcRow = (y + dy) * image.Width;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dy == 0 && dx == 0) continue;
                    T pixelVal = buffer[srcRow + x + dx];
                    descriptor[index] = pixelVal >= centerVal;
                    index++;
                }
            }

            descriptors.Add(descriptor);
        }
    }

    public static List<CVDescriptor> CensusFeatures(CVImage image, int radius, int offsetX, int y)
    {
        List<CVDescriptor> descriptors = new List<CVDescriptor>();

        if (image.DataFormat == CVDataFormat.CV_U8) censusFeatures<byte>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S8) censusFeatures<sbyte>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U16) censusFeatures<ushort>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S16) censusFeatures<short>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U32) censusFeatures<uint>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S32) censusFeatures<int>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_U64) censusFeatures<ulong>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_S64) censusFeatures<long>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F32) censusFeatures<float>(image, radius, offsetX, y, ref descriptors);
        else if (image.DataFormat == CVDataFormat.CV_F64) censusFeatures<double>(image, radius, offsetX, y, ref descriptors);

        return descriptors;
    }

    public static void MatchFeaturesCensus(
        CVImage image1,
        CVImage image2,
        int radiusCensus,
        int xOffset,
        int yOffset,
        out List<(int x, int y)> matchedFeatures1,
        out List<(int x, int y)> matchedFeatures2)
    {
        matchedFeatures1 = new List<(int x, int y)>();
        matchedFeatures2 = new List<(int x, int y)>();

        // Scan image line and create features
        for (int y = 0; y < image1.Height; y += yOffset)
        {
            List<CVDescriptor> features1 = CensusFeatures(image1, radiusCensus, xOffset, y);
            List<CVDescriptor> features2 = CensusFeatures(image2, radiusCensus, 1, y);

            if (features1.Count == 0 || features2.Count == 0)
                continue;

            for (int i = 0; i < features1.Count; i++)
            {
                int bestIndex = -1;
                int bestDistance = int.MaxValue;

                for (int j = 0; j < features2.Count; j++)
                {
                    int distance = features1[i].Distance(features2[j]);

                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestIndex = j;
                    }
                }

                matchedFeatures1.Add(features1[i].Position);
                matchedFeatures2.Add(features2[bestIndex].Position);
            }
        }
    }

    private static readonly int[,] directionsForward = {
        { 1, 0},
        { 0, 1},
        { 1, 1},
        {-1, 1},
    };


    private static readonly int[,] directionsBackward = {
        {-1, 0},
        { 0,-1},
        {-1,-1},
        { 1,-1},
    };

    private static int computeBlockCost<T>(CVImage left, CVImage right, int width, int height, int x, int y, int disparity, int radius, int PInvalid) where T : struct, INumber<T>
    {
        Span<T> leftBuffer = left.BufferAs<T>();
        Span<T> rightBuffer = right.BufferAs<T>();

        int cost = 0;
        int minX = Math.Max(0, x - radius);
        int maxX = Math.Min(width - 1, x + radius);
        int minY = Math.Max(0, y - radius);
        int maxY = Math.Min(height - 1, y + radius);

        for (int yy = minY; yy <= maxY; yy++)
        {
            int row = yy * width;
            for (int xx = minX; xx <= maxX; xx++)
            {
                int rightX = xx - disparity;

                /* * No corresponding pixel exists in the right image. */
                if (rightX < 0 || rightX >= width)
                    cost += PInvalid;
                else
                {
                    int leftValue = int.CreateChecked(leftBuffer[row + xx]);
                    int rightValue = int.CreateChecked(rightBuffer[row + rightX]);

                    cost += Math.Abs(leftValue - rightValue);
                }
            }
        }
        return cost;
    }

    private static int getMin(Span<int> list)
    {
        int min = list[0];

        for (int i = 1; i < list.Length; i++)
        {
            int element = list[i];
            if (element < min)
                min = element;
        }

        return min;
    }

    private static int calculateBest(Span<int> previous, int minPrevious, int maxDisparity, int disparity, int P1, int P2)
    {
        /* * Same disparity. */
        int best = previous[disparity];

        /* * Disparity - 1. */
        if (disparity > 0)
        {
            int candidate = previous[disparity - 1] + P1;

            if (candidate < best)
                best = candidate;
        }

        /* * Disparity + 1. */
        if (disparity + 1 < maxDisparity)
        {
            int candidate = previous[disparity + 1] + P1;

            if (candidate < best)
                best = candidate;
        }

        /* * Any other disparity. */
        int largeJump = minPrevious + P2;

        if (largeJump < best) best = largeJump;

        return best;
    }

    private static void processForwardDirections<T>(int width, int height, CVImage[] costs, int[,] ds, int maxDisparity, int blockSize, int P1, int P2, int PInvalid, int[] aggregatedCost) where T : struct, INumber<T>
    {
        int directions = ds.GetLength(0);

        int[] previousRow = new int[width * maxDisparity * directions];
        int[] currentRow = new int[width * maxDisparity * directions];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int xOffset = x * maxDisparity * directions;

                int pixelIndex = y * width + x;
                int aggregateBase = pixelIndex * maxDisparity;

                for (int disparity = 0; disparity < maxDisparity; disparity++)
                {
                    Span<T> costBuffer = costs[disparity].BufferAs<T>();

                    int cost = int.CreateChecked(costBuffer[x * y * width]); // computeBlockCost<T>(left, right, width, height, x, y, disparity, blockSize, PInvalid);
                    for (int i = 0; i < directions; i++)
                    {
                        int previousX = x - ds[i, 0];
                        int previousY = y - ds[i, 1];

                        int directionOffset = i * maxDisparity;
                        int xDirectionOffset = xOffset + directionOffset;

                        int currentIndex = xDirectionOffset + disparity;

                        // No Previous Pixels
                        if (previousX < 0 || previousX >= width || previousY < 0 || previousY >= height)
                        {
                            currentRow[currentIndex] = cost;
                        }
                        else
                        {
                            int previousXOffset = previousX * maxDisparity * directions;
                            int startIndex = previousXOffset + directionOffset;

                            Span<int> previousScores;
                            if (ds[i, 1] == 0) previousScores = currentRow.AsSpan(startIndex, maxDisparity);
                            else previousScores = previousRow.AsSpan(startIndex, maxDisparity);

                            int minPrevious = getMin(previousScores);
                            int best = calculateBest(previousScores, minPrevious, maxDisparity, disparity, P1, P2);

                            currentRow[currentIndex] = cost + best - minPrevious;
                        }

                        aggregatedCost[aggregateBase + disparity] += currentRow[currentIndex];
                    }
                }
            }

            int[] temp = previousRow;
            previousRow = currentRow;
            currentRow = temp;
        }
    }

    private static void processBackwardDirections<T>(int width, int height, CVImage[] costs, int[,] ds, int maxDisparity, int blockSize, int P1, int P2, int PInvalid, int[] aggregatedCost) where T : struct, INumber<T>
    {
        int directions = ds.GetLength(0);

        int[] previousRow = new int[width * directions * maxDisparity];
        int[] currentRow = new int[width * directions * maxDisparity];

        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = width - 1; x >= 0; x--)
            {
                int xOffset = x * maxDisparity * directions;

                int pixelIndex = y * width + x;
                int aggregateBase = pixelIndex * maxDisparity;

                for (int disparity = 0; disparity < maxDisparity; disparity++)
                {
                    Span<T> costBuffer = costs[disparity].BufferAs<T>();

                    int cost = int.CreateChecked(costBuffer[x * y * width]); // computeBlockCost<T>(left, right, width, height, x, y, disparity, blockSize, PInvalid);

                    for (int i = 0; i < directions; i++)
                    {
                        int previousX = x - ds[i, 0];
                        int previousY = y - ds[i, 1];

                        int directionOffset = i * maxDisparity;
                        int xDirectionOffset = xOffset + directionOffset;

                        int currentIndex = xDirectionOffset + disparity;

                        // No Previous Pixels
                        if (previousX < 0 || previousX >= width || previousY < 0 || previousY >= height)
                        {
                            currentRow[currentIndex] = cost;
                        }
                        else
                        {
                            int previousXOffset = previousX * maxDisparity * directions;
                            int startIndex = previousXOffset + directionOffset;

                            Span<int> previousScores;
                            if (ds[i, 1] == 0) previousScores = currentRow.AsSpan(startIndex, maxDisparity);
                            else previousScores = previousRow.AsSpan(startIndex, maxDisparity);

                            int minPrevious = getMin(previousScores);
                            int best = calculateBest(previousScores, minPrevious, maxDisparity, disparity, P1, P2);

                            currentRow[currentIndex] = cost + best - minPrevious;
                        }

                        aggregatedCost[aggregateBase + disparity] += currentRow[currentIndex];
                    }
                }
            }

            int[] temp = previousRow;
            previousRow = currentRow;
            currentRow = temp;
        }
    }

    public static void ComputeSGBM<T>(
        CVImage left,
        CVImage right,
        ref CVImage disparity,
        int maxDisparity = 256,
        int blockRadius = 5,
        int P1 = 10,
        int P2 = 120,
        int PInvalid = 1000)
        where T : struct, INumber<T>, IMinMaxValue<T>
    {
        int width = left.Width;
        int height = left.Height;

        CVImage[] costs = new CVImage[maxDisparity];

        for (int i = 0; i < maxDisparity; i++)
        {
            CVImage shifted = CVShift.Shift(right, i, 0);
            CVImage absoluteDifference = CVMath.Abs(left - shifted);
            costs[i] = CVWindowing.SumWindow(absoluteDifference, blockRadius);
        }

        int[] aggregatedCost = new int[width * height * maxDisparity];

        processForwardDirections<T>(width, height, costs, directionsForward, maxDisparity, blockRadius, P1, P2, PInvalid, aggregatedCost);
        processBackwardDirections<T>(width, height, costs, directionsBackward, maxDisparity, blockRadius, P1, P2, PInvalid, aggregatedCost);

        Span<T> disparityBuffer = disparity.BufferAs<T>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = y * width + x;
                int baseIndex = pixelIndex * maxDisparity;

                int bestDisparity = -1;
                int bestCost = int.MaxValue;
                int secondBestCost = int.MaxValue;

                for (int d = 0; d < maxDisparity; d++)
                {
                    int cost = aggregatedCost[baseIndex + d];

                    if (cost < bestCost)
                    {
                        secondBestCost = bestCost;
                        bestCost = cost;
                        bestDisparity = d;
                    }
                    else if (cost < secondBestCost)
                    {
                        secondBestCost = cost;
                    }
                }

                if (bestDisparity < 0)
                {
                    disparityBuffer[pixelIndex] = T.Zero;
                    continue;
                }

                disparityBuffer[pixelIndex] = T.CreateChecked(bestDisparity);
            }
        }
    }


    public static CVImage ComputeSGBM(CVImage image1, CVImage image2, int maxDisparity = 64,
        int blockSize = 5,
        int P1 = 10,
        int P2 = 120)
    {
        CVImage outImage = CVImage.Create(image1.Width, image1.Height, image1.DataFormat, image1.ChannelFormats);

        if (image1.DataFormat == CVDataFormat.CV_U8) ComputeSGBM<byte>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_S8) ComputeSGBM<sbyte>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_U16) ComputeSGBM<ushort>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_S16) ComputeSGBM<short>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_U32) ComputeSGBM<uint>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_S32) ComputeSGBM<int>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_U64) ComputeSGBM<ulong>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_S64) ComputeSGBM<long>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_F32) ComputeSGBM<float>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);
        else if (image1.DataFormat == CVDataFormat.CV_F64) ComputeSGBM<double>(image1, image2, ref outImage, maxDisparity, blockSize, P1, P2);

        return outImage;
    }

    public static List<(int, int, double)> DetectFeatureFast(
        CVImage image)
    {
        CVImage gray = CVConvert.ConvertChannelFormat(image, CVChannelFormat.CV_Grayscale);

        List<(int, int)> fastKeypoints = Fast(gray, 5.0);
        List<(int, int, double)> fastHarrisScores = CVCornerDetector.HarrisStrengthPoints(gray, fastKeypoints, 0.04, 3);
        List<(int, int, double)> supressedPoints = CVCornerDetector.NonMaximumSuppressionPoints(fastHarrisScores, 3);
        List<(int, int, double)> bestPoints = supressedPoints.Take(500).ToList();

        return bestPoints;
    }
}