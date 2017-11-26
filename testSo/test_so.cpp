#include<string>
#include<iostream>
#include "inference.h"
#include<fstream>
#include<sstream>
#include<time.h>
#include<thread>
#include<vector>
int main(int argc, char* argv[])
{
  int img[299*299*3];
  for(int i=0 ; i<299*299*3 ; i++)
  {
	  img[i] = 50;
  }
  Sess* sess=start_sess("/xhome/tx_zhiwei/facial_age_results/freeze_test/age_model.pb");
  float result[89+1];
  inference(sess, img, 299, 299, result, 90);
  for(int i=0; i<90; i++)
    std::cout<<i<<" "<<result[i]<<std::endl;
  close_sess(sess);
  return 0;
}
