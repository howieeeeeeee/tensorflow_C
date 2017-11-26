bazel build tensorflow/python/tools:freeze_graph && \
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/xhome/tx_zhiwei/facial_age_results/freeze_test/age_graph_test.pb \
--input_checkpoint=/xhome/tx_zhiwei/facial_age_results/freeze_test/model.ckpt-99999 \
--output_graph=/xhome/tx_zhiwei/facial_age_results/freeze_test/age_model_test.pb \
--output_node_names=inception_v3/logits/predictions
  