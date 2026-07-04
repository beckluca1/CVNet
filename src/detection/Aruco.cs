namespace CVNet;

public class CVAruco
{
    public static Dictionary<ulong, int> Aruco_6x6_50 = new Dictionary<ulong, int>
    {
        {8117912230,0},
        {4021958801,1},
        {5788658381,2},
        {53984036510,3},
        {57453407765,4},
        {58226249320,5},
        {17826529781,6},
        {36680299162,7},
        {13016311037,8},
        {16155626300,9},
        {18756695267,10},
        {19554218583,11},
        {30338813894,12},
        {36202065159,13},
        {37969630198,14},
        {43680128222,15},
        {2681334212,16},
        {5718659471,17},
        {12895588578,18},
        {19335674621,19},
        {23319354806,20},
        {27523229516,21},
        {31919486849,22},
        {41426984179,23},
        {45578993700,24},
        {53273203856,25},
        {52028803393,26},
        {62084121899,27},
        {62864009288,28},
        {62650021809,29},
        {67165901487,30},
        {1707079613,31},
        {1411216086,32},
        {3480372898,33},
        {5159664107,34},
        {5813911156,35},
        {15637933726,36},
        {21225154240,37},
        {22294523168,38},
        {23785877044,39},
        {26853935627,40},
        {25857337489,41},
        {26131589562,42},
        {28990475643,43},
        {30246791759,44},
        {31545816858,45},
        {32905466620,46},
        {35980271786,47},
        {37091560511,48},
        {39706921748,49},
    };

    public static List<CVContour> GetMarker(CVImage image, int markerSize)
    {
        CVImage gray = CVConvert.ToFormat(image, CVChannelFormat.CV_Grayscale);
        CVImage thresh = CVProcessing.AdaptiveThresholdMean(gray, 7, 13);

        List<CVContour> contours = CVDetection.FindContours(thresh, 1);
        List<CVContour> checkedContours = CVDetection.ContourChecks(contours, image.Width, image.Height);
        List<CVContour> polygons = CVDetection.ApproximatePolygons(checkedContours, 0.01);
        List<CVContour> quads = CVDetection.QuadrilateralChecks(polygons, image.Width, image.Height);
        List<int> contourGroups = CVDetection.GroupContours(quads, image.Width, image.Height, markerSize);

        if (quads.Count > 0) Console.WriteLine(quads.Count);

        List<CVContour> validContours = CVDetection.IdentifyContours(gray, quads, contourGroups, markerSize);
        for (int i = 0; i < validContours.Count; i++) Console.WriteLine($"Valid Contour {i}: {validContours[i].ID}");

        return validContours;
    }
}