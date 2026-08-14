namespace CVNet;

public class CVImagePyramid
{
    public int Levels = 0;

    private Dictionary<int, CVImage> imageList;

    public CVImagePyramid(CVImage image, int levels, double scaling)
    {
        Levels = levels;

        imageList = new Dictionary<int, CVImage>();

        CVImage currentImage = image;
        imageList.Add(0, currentImage);

        for (int i = 1; i < Levels; i++)
        {
            currentImage = CVResize.StretchLinear(currentImage, currentImage.Width / 2, currentImage.Height / 2);
            imageList.Add(i, currentImage);
        }
    }

    public CVImagePyramid(int levels)
    {
        Levels = levels;

        imageList = new Dictionary<int, CVImage>();

        for (int i = 1; i < Levels; i++)
            imageList.Add(i, CVImage.Create());
    }

    public CVImage this[int index]
    {
        get => imageList[index];
        set => imageList[index] = value;
    }

    public static CVImagePyramid operator +(int val, CVImagePyramid image) { return CVMath.Add(image, val); }
    public static CVImagePyramid operator +(double val, CVImagePyramid image) { return CVMath.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image, int val) { return CVMath.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image, double val) { return CVMath.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image1, CVImagePyramid image2) { return CVMath.Add(image1, image2); }

    // public static  CVImagePyramid operator -(int val, CVImagePyramid image) { return CVMath.Subtract(val, image); }
    // public static  CVImagePyramid operator -(double val, CVImagePyramid image) { return CVMath.Subtract(val, image); }
    public static CVImagePyramid operator -(CVImagePyramid image, int val) { return CVMath.Subtract(image, val); }
    public static CVImagePyramid operator -(CVImagePyramid image, double val) { return CVMath.Subtract(image, val); }
    public static CVImagePyramid operator -(CVImagePyramid image1, CVImagePyramid image2) { return CVMath.Subtract(image1, image2); }

    public static CVImagePyramid operator *(int val, CVImagePyramid image) { return CVMath.Multiply(image, val); }
    public static CVImagePyramid operator *(double val, CVImagePyramid image) { return CVMath.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image, int val) { return CVMath.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image, double val) { return CVMath.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image1, CVImagePyramid image2) { return CVMath.Multiply(image1, image2); }

    // public static  CVImagePyramid operator /(int val, CVImagePyramid image) { return CVMath.Divide(val, image); }
    // public static  CVImagePyramid operator /(double val, CVImagePyramid image) { return CVMath.Divide(val, image); }
    public static CVImagePyramid operator /(CVImagePyramid image, int val) { return CVMath.Divide(image, val); }
    public static CVImagePyramid operator /(CVImagePyramid image, double val) { return CVMath.Divide(image, val); }
    public static CVImagePyramid operator /(CVImagePyramid image1, CVImagePyramid image2) { return CVMath.Divide(image1, image2); }

}