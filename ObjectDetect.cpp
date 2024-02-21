#include <stdio.h>
#include "my_blobs.h"

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("usage: ObjectDetect <Image_Path> LowerThreshold UpperThreshold\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    int lower_thresh = atoi(argv[2]); //Hue range - low threshold
    int upper_thresh = atoi(argv[3]); //Hue range - high threshold
    if ( lower_thresh >= upper_thresh )
    {
        printf("Lower threshold must be < upper one \n");
        return -1;
    }

    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);

    //SMOOTHING FILTER
    Mat filtered;
    Mat mask = (Mat_<float>(3,3) <<  1, 1, 1,
                                     1, 1, 1,
                                     1, 1, 1);
    float norm = 9.0; //normalization factor for the mask above
    filter2D(image, filtered, -1, mask / norm); //apply normalization factor

    //CONVERT TO HSV, SEPARATE CHANNELS AND SHOW EACH ONE
    Mat img_HSV, imgHSV[3]; //HSV image and separated channels
    cvtColor(filtered, img_HSV, COLOR_BGR2HSV);
    split(img_HSV, imgHSV); //imgHSV contains separated channels

    Mat temp, ThrH, ThrS, SelectedColour;
    // --ALEX -- ask user to insert saturation
    int lowerSat, upperSat;
    printf("Please insert lower and upper Saturation Threshold:");
    scanf("%d %d", &lowerSat, &upperSat);
    int maxval = 255; //binary output High
    inRange(imgHSV[0], lower_thresh, upper_thresh, ThrH); //Get pixels in Hue colour range
      namedWindow("Hue Thresholded", WINDOW_NORMAL );
      imshow("Hue Thresholded", ThrH); //Hue
    inRange(imgHSV[1], lowerSat, upperSat, ThrS); //Use Sat to keep only pixels with dominant colour
      namedWindow("Saturation Thresholded", WINDOW_NORMAL );
      imshow("Saturation Thresholded", ThrS); //Hue
      waitKey(0);
    //logic AND both thresholded images
    bitwise_and(ThrH, ThrS, SelectedColour);
      namedWindow("Final color highlighting", WINDOW_AUTOSIZE );
      imshow("Final color highlighting", SelectedColour); //Selected colour pixels
      waitKey(0);
      destroyWindow("Hue Thresholded");
      destroyWindow("Saturation Thresholded");

    //  exit(0); //STEP 1 - COMMENT TO CONTINUE PROGRAM

     //OPENING operation to remove tiny regions
     Mat OpenedImg;
     // Create mask element for the erode/dilate process as a 3x3 pix square
     // this will kill regions with any dimension smaller than 6 pixels
     Mat erode_mask = getStructuringElement(MORPH_RECT,Size(3,3));
    //  dilate(SelectedColour,OpenedImg,erode_mask);
    //  erode(OpenedImg,OpenedImg,erode_mask);
     erode(SelectedColour,OpenedImg,erode_mask);
     dilate(OpenedImg,OpenedImg,erode_mask);
      imshow("Opened image", OpenedImg); //Tiny regions removed
      waitKey(0);
     destroyWindow("Final color highlighting");

    // exit(0); //STEP 2 - COMMENT TO CONTINUE PROGRAM

     //Do labelling and extract regions locations
     Mat labelled_img;
     vector<int> areas;
     vector<int> labels;
     vector<Rect> locations;
     vector<Point2f> centroids;
     //Do it on initial selected colour pixels image
     int n=MyConnectedComponentsWithStats(SelectedColour,labelled_img, areas, locations, centroids, labels);
     printf("SelectedColour image: found %d blobs\n",n);
     areas.clear();
     locations.clear();
     centroids.clear();
     //Do it on cleaned image
     bool showLabelledImg=true;
     printf("Showing cleaned image labelling\n");
     n=MyConnectedComponentsWithStats(OpenedImg,labelled_img, areas, locations, centroids, labels, showLabelledImg);
     printf("OpenedImg image: found %d blobs\n",n);
     // Show basic Statistics
     printf("Blob\tLabel\tArea\tXmin\tYmin\tWidth\tHeight\tXcog\tYcog\n");
     for(int i=0; i<n; i++)
     {
      Rect r = locations[i];
      Point2f p = centroids[i];
      printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\n",i,labels[i],areas[i],r.x,r.y,r.width,r.height,p.x,p.y);
     }

    //NOW USE THE INFORMATION ABOUT THE BLOBS (area, locations, centroids) TO KEEP ONLY THE BOX!
    //YOUR CODE HERE

    
    int area_threshold = 500;  
    int width_threshold = 50; 

    for (int i = 0; i < n; i++)
    {
        Rect r = locations[i];
        Point2f p = centroids[i];

        // Check if the blob meets criteria
        if (areas[i] >= area_threshold || r.width >= width_threshold) 
        {
            printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\n", i, labels[i], areas[i], r.x, r.y, r.width, r.height, p.x, p.y);

            // Optional: Draw a rectangle around the blob on a copy of the image
            Mat displayImage = image.clone();  // Create a copy if you want to draw
            rectangle(displayImage, r, Scalar(0, 255, 0), 2); // Green rectangle
            imshow("Filtered Blobs", displayImage); 
            waitKey(0); 
        }
    }

    return 0;
}
