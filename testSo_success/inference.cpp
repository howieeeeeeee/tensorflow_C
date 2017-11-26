#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <Sess.hpp>
#include <string>

Sess* start_sess(std::string pb_path)
{
  try
  {
    Sess* sess=new Sess(pb_path);
    return sess;
  }
  catch(const Sess_Init_Exception& e)
  {
    std::cerr<<e.what()<<std::endl; 
    Sess* sess = nullptr;
    return sess;
  }
}
bool inference(Sess* sess, int* image_data, int rows, int cols, float* result, int result_kinds) {

  std::cout<<"hello world"<<std::endl;
  if(sess == nullptr)
  {
    return false;
  }
  
  tensorflow::Tensor img_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({rows, cols, 3}));
  auto map=img_tensor.flat<int,3>();
  
  //memcpy(map.data(), image_data, cols*rows*3*sizeof(int));
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      map(i, j, 0) = 50;
      map(i, j, 1) = 50;
      map(i, j, 2) = 50;
    }
  }
  
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
    { "input_node:0", img_tensor }, };
  std::vector<tensorflow::Tensor> outputs;
  sess->sess()->Run(inputs,{"inception_v3/logits/predictions:0"},{}, &outputs);
  
  auto output_map=outputs[0].flat<float>();
  memcpy(result, output_map.data(), result_kinds*sizeof(float));
  return true;
}
void close_sess(Sess* sess)
{
  if(sess!=nullptr)
  {
    delete sess;
    sess=nullptr;
  }
}
