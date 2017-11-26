#include<string>
class Sess;
Sess* start_sess(std::string pb_path);
int inference(Sess* sess, int* img_data, int width, int height, float* result, int result_kinds);
int close_sess(Sess* sess);     
