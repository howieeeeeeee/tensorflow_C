#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{

   Mat image;
   image = imread("/xhome/tx_zhiwei/inceptionv3_center_loss/test_pic/liang.jpg");
   FILE *fp_1; 
   FILE *fp_2; 
   FILE *fp_3; 
   fp_1=fopen("mat_1.txt","w");
   fp_2=fopen("mat_2.txt","w");
   fp_3=fopen("mat_3.txt","w");
   
   cout<<image.rows<<endl;
   cout<<image.cols<<endl;
   for(int i = 0;i<image.rows;i++)
   {
     for(int j = 0;j<image.cols;j++)
     {
       fprintf(fp_1,"%d",image.at<Vec3b>(i, j)[0]);//B
       fprintf(fp_2,"%d",image.at<Vec3b>(i, j)[1]);//G
       fprintf(fp_3,"%d",image.at<Vec3b>(i, j)[2]);//R
       fprintf(fp_1," ");
       fprintf(fp_2," ");
       fprintf(fp_3," ");
     }
     fprintf(fp_1,"\n");
     fprintf(fp_2,"\n");
     fprintf(fp_3,"\n");
   }
   
   return 0;
}