cc_binary(name = "libinference.so", 
          srcs = ["inference.cpp", "Sess.hpp"],
          linkopts = ["-shared"],
          linkshared = 1,
          includes = ["."],
          deps = ["//tensorflow/core:tensorflow", ],
)

cc_binary(name = "inferrence",
          srcs = ["inference.cpp", "Sess.hpp"],
          includes = ["."],
          deps = ["//tensorflow/core:tensorflow", ],)

cc_binary(name = "libinference.a",
          srcs = ["inference.cpp", "Sess.hpp"],
          linkstatic = 1,
          includes = ["."],
          deps = ["//tensorflow/core:tensorflow", ],)
