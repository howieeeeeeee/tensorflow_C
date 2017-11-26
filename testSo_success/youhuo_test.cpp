#include <string>
#include <iostream>
#include "inference.h"
#include <fstream>
#include <sstream>
#include <time.h>
#include <thread>
#include <vector>
#include <stdio.h>
//#include <opencv2/opencv.hpp>

//using namespace cv;

int main(int argc, char** argv ) {   
    
    int img[500*500*3];
    std::ifstream ifs("./tt.txt");
    for(int i=0 ; i<500*500*3 ; i++) {       
         std::string str;
            ifs>>str;
            std::istringstream(str)>>img[i];
    }
    ifs.close();
    
    Sess* sess=start_sess("age_model_test.pb");
    float result[89];
    inference(sess, img, 500, 500, result, 88+1);
    for (int i = 0; i < 89; i++) {
        std::cout<<result[i]<<std::endl;
    }
   
    return 0;
}

/*
int main(int argc, char* argv[])
{
  int img[500*340*3];
  std::cout<<"end:"<<"dasd"<<std::endl;
  std::ifstream ifs("./pixel.txt");
  for(int i=0 ; i<500*340*3 ; i++)
  {
	  std::string str;
	  ifs>>str;
	  std::istringstream(str)>>img[i];
          //std::cout<<img[i]<<std::endl;
  }
  ifs.close();
  Sess* sess=start_sess("affective_frozen_graph.pb");
  float result[5][7+1];
  int threads=4;
  std::vector<int> ranges;
  ranges.push_back(0); ranges.push_back(25); ranges.push_back(50); ranges.push_back(75); ranges.push_back(100);
  std::vector<std::thread> vthread;
  for(int i=0; i<threads; i++)
  {
	vthread.push_back(std::thread([&,i](){
	std::cout<<"begin:"<<i<<std::endl;
	for(int j=ranges[i]; j<ranges[i+1]; j++)
	{
	  inference(sess, img, 340, 500, result[j], 763+1);
          std::cout<<"i:"<<i<<std::endl;	  
	}
	std::cout<<"end:"<<i<<std::endl;
	}));
  }
  for(int i=0; i<4; i++)
  {
	  vthread[i].join();
  }
  for(int i=0; i<100; i++)
  {
    for(int j=0; j<763+1; j++)
    {
      if(result[0][j]!=result[i][j])
      {
	 std::cerr<<i<<" "<<j<<" "<<result[0][j]<<" "<<result[i][j]<<std::endl;
      }
    }
  }
  return 0;
}
*/
