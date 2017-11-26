#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <string>
#include <stdexcept>
class Sess_Init_Exception:public std::runtime_error
{
public:
  Sess_Init_Exception(const std::string& error_msg):std::runtime_error(error_msg)
  {
    
  }
};
class Sess
{
public:
  Sess(std::string pb_path)
  {
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
      throw Sess_Init_Exception(status.ToString());
    }

    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), pb_path, &graph_def);
    if (!status.ok()) {
      throw Sess_Init_Exception(status.ToString());
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
      throw Sess_Init_Exception(status.ToString());
    }

  }
  tensorflow::Session* session;
  tensorflow::Session* sess()
  {
    return session;
  }
  ~Sess()
  {
    session->Close();
  }
};
