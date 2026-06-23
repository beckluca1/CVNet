using System.Numerics;

namespace CVNet;

public static class CVContour
{
    private static readonly byte BACKGROUND = 0;
    private static readonly byte VISITED_OUTER_RIGHT = 100;
    private static readonly byte VISITED = 200;
    private static readonly byte FOREGROUND = 255;

    private static readonly int[] dx = new int[] { 1, 1, 0, -1, -1, -1, 0, 1 };
    private static readonly int[] dy = new int[] { 0, -1, -1, -1, 0, 1, 1, 1 };

    private static void traceExternalContourMock(byte[] imageState, int startX, int startY, int width, int height)
    {
        int startIndex = startX + width * startY;

        int currentX = startX;
        int currentY = startY;
        int currentIndex = startIndex;

        int startDir = -1;
        int searchIdx = 5;


        int dir = -1;

        bool is_first_move = true;
        // 3. TRACING LOOP
        while (true)
        {
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = searchIdx + n;
                int nextDir = idx & 7;
                int nextX = currentX + dx[nextDir];
                int nextY = currentY + dy[nextDir];
                int nextIndex = nextX + nextY * width;
                if (imageState[nextIndex] == BACKGROUND) continue;
                if (((searchIdx <= 1) || (dir <= searchIdx - 2)) && (currentX != startX && currentY != startY))//do nt apply to first pixel in the way back
                    imageState[currentIndex] = VISITED_OUTER_RIGHT;

                // --- EXECUTE MOVE ---
                dir = nextDir;
                currentY = nextY;
                currentX = nextX;

                if (currentY < 1) return;

                // Short-circuit Jacob's Check
                if (currentIndex == startIndex)
                {
                    if (!is_first_move && dir == startDir)
                    {
                        return;//done
                    }
                }

                currentIndex = nextIndex;

                // Reset search index for Moore neighbor
                searchIdx = (dir + 6) & 7;
                break;

            }
            if (is_first_move)
            {
                if (dir == -1)
                {//single pixel
                    break;//not moved
                }
                startDir = dir;
                is_first_move = false;
            }

        }
    }

    private static bool traceContour(byte[] imageState, List<(int, int)> contour, int startX, int startY, int width, int height, bool isExternal)
    {
        int startIndex = startX + width * startY;

        int currentX = startX;
        int currentY = startY;
        int currentIndex = startIndex;

        int start_dir = -1;
        int search_idx = isExternal ? 5 : 1;

        int dir = -1;
        // int sign=isExternal?1:-1;

        bool is_first_move = true;

        // QUick test to find the first element of an internal contour. We know it should be NE
        if (!isExternal)
        {
            int n = 0;
            for (n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                int nextDir = idx & 7;
                int nextX = currentX + dx[nextDir];
                int nextY = currentY + dy[nextDir];
                int nextIndex = nextX + nextY * width;
                if (imageState[nextIndex] == BACKGROUND) continue;
                if (nextX != startX + 1 || nextY != startY - 1) return false;
                break;
            }
            if (n == 8) return false;//isolated pixels must not be considered as internal
        }
        // 3. TRACING LOOP
        for (int c = 0; c < 10000; c++)
        {
            contour.Add((currentX - 1, currentY - 1));
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                int nextDir = idx & 7;
                int nextX = currentX + dx[nextDir];
                int nextY = currentY + dy[nextDir];
                int nextIndex = nextX + nextY * width;
                if (imageState[nextIndex] == BACKGROUND) continue;

                // --- EXECUTE MOVE ---
                dir = nextDir;
                currentY = nextY;
                currentX = nextX;

                if (currentY < 1) return false;

                if ((search_idx <= 1) || (dir <= search_idx - 2))
                {
                    imageState[currentIndex] = VISITED_OUTER_RIGHT;
                }
                else if (imageState[currentIndex] == FOREGROUND)
                {
                    imageState[currentIndex] = VISITED;
                }

                // Short-circuit Jacob's Check
                if (currentIndex == startIndex)
                {
                    if (!is_first_move && dir == start_dir)
                    {
                        return true;//done
                    }
                }

                currentIndex = nextIndex;

                // Reset search index for Moore neighbor
                search_idx = (dir + 6) & 7;
                break;

            }
            if (is_first_move)
            {
                if (dir == -1)
                {//single pixel
                    imageState[currentIndex] = VISITED_OUTER_RIGHT;
                    break;//not moved
                }
                start_dir = dir;
                is_first_move = false;
            }

        }
        return true;
    }

    public static int findStartContourPoint(byte[] imageState, int o, int j, int width, int height)
    {
        for (; j < width && !(imageState[o + j] != BACKGROUND); ++j) ;
        return j;
    }

    public static int findEndContourPoint(byte[] imageState, int o, int j, int width, int height)
    {
        for (; j < width && (imageState[o + j] != BACKGROUND); ++j) ;
        return j;
    }

    public static List<List<(int, int)>> TraceContours<T1, T2>(CVImage image, T2 foreground, ref List<List<(int, int)>> contours) where T1 : struct, IEqualityOperators<T1, T1, bool> where T2 : struct
    {
        Span<T1> buffer = image.BufferAs<T1>();

        T1 fg = (T1)Convert.ChangeType(foreground, typeof(T1));

        int width = image.Width + 2;
        int height = image.Height + 2;

        byte[] imageState = new byte[width * height];

        int widthDiff = width - image.Width;

        int srcI = 0;
        int dstI = 1 + 1 * width;
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                imageState[dstI] = (buffer[srcI] == fg) ? FOREGROUND : BACKGROUND;
                srcI++;
                dstI++;
            }
            dstI += widthDiff;
        }

        for (int y = 1; y <= height - 1; ++y)
        {
            int yOffset = y * width;

            // "c" is updated by the find* functions
            for (int x = 1; x < width - 1;)
            {
                // 1. FAST SCAN: Skip background pixels
                if ((x = findStartContourPoint(imageState, yOffset, x, width, height)) == width) break;

                // 2. CHECK: Only process if actually FOREGROUND (redundancy check)
                if (imageState[yOffset + x] == FOREGROUND && y < height - 1)
                {
                    List<(int, int)> contour = new List<(int, int)>();
                    if (traceContour(imageState, contour, x, y, width, height, true))
                    {
                        contours.Add(contour);
                    }
                }
                else if (imageState[yOffset + x] == FOREGROUND && y == height - 1)
                {//extra step to mark east pixels only so that internal contours can be correctly extracted in this line
                    traceExternalContourMock(imageState, x, y, width, height);
                }

                // 3. FAST SCAN: Find end of current component to skip processing it again
                x = findEndContourPoint(imageState, yOffset, x + 1, width, height);
                if (x >= width) break;//end of row
                                      //internal contour
                if (imageState[yOffset + x - 1] > VISITED_OUTER_RIGHT && y > 1)
                {//inner contours of first line are handled by the thread above

                    List<(int, int)> contour = new List<(int, int)>();
                    if (traceContour(imageState, contour, x - 1, y, width, height, false))
                    {
                        contours.Add(contour);
                    }
                }
            }
        }
        return contours;
    }
}