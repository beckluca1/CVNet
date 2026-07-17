namespace CVNet;

public class CVImagePyramid
{
    public int Levels = 0;

    private Dictionary<int, CVImage> imageList;

    public CVImagePyramid(CVImage image, int levels)
    {
        Levels = levels;

        imageList = new Dictionary<int, CVImage>();

        CVImage currentImage = image;
        imageList.Add(0, currentImage);

        for (int i = 1; i < Levels; i++)
        {
            currentImage = CVResize.Resize(currentImage, currentImage.Width / 2, currentImage.Height / 2, CV_ResizeMode.CV_CROP_NEAREST);
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

    public static CVImagePyramid operator +(int val, CVImagePyramid image) { return CVAdd.Add(image, val); }
    public static CVImagePyramid operator +(double val, CVImagePyramid image) { return CVAdd.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image, int val) { return CVAdd.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image, double val) { return CVAdd.Add(image, val); }
    public static CVImagePyramid operator +(CVImagePyramid image1, CVImagePyramid image2) { return CVAdd.Add(image1, image2); }

    // public static  CVImagePyramid operator -(int val, CVImagePyramid image) { return CVSubtract.Subtract(val, image); }
    // public static  CVImagePyramid operator -(double val, CVImagePyramid image) { return CVSubtract.Subtract(val, image); }
    public static CVImagePyramid operator -(CVImagePyramid image, int val) { return CVSubtract.Subtract(image, val); }
    public static CVImagePyramid operator -(CVImagePyramid image, double val) { return CVSubtract.Subtract(image, val); }
    public static CVImagePyramid operator -(CVImagePyramid image1, CVImagePyramid image2) { return CVSubtract.Subtract(image1, image2); }

    public static CVImagePyramid operator *(int val, CVImagePyramid image) { return CVMultiply.Multiply(image, val); }
    public static CVImagePyramid operator *(double val, CVImagePyramid image) { return CVMultiply.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image, int val) { return CVMultiply.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image, double val) { return CVMultiply.Multiply(image, val); }
    public static CVImagePyramid operator *(CVImagePyramid image1, CVImagePyramid image2) { return CVMultiply.Multiply(image1, image2); }

    // public static  CVImagePyramid operator /(int val, CVImagePyramid image) { return CVDivide.Divide(val, image); }
    // public static  CVImagePyramid operator /(double val, CVImagePyramid image) { return CVDivide.Divide(val, image); }
    public static CVImagePyramid operator /(CVImagePyramid image, int val) { return CVDivide.Divide(image, val); }
    public static CVImagePyramid operator /(CVImagePyramid image, double val) { return CVDivide.Divide(image, val); }
    public static CVImagePyramid operator /(CVImagePyramid image1, CVImagePyramid image2) { return CVDivide.Divide(image1, image2); }

}