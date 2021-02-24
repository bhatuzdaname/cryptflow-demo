import tensorflow as tf
import numpy as np

import sys

def load_pb(path_to_pb):
  with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    return graph


def run(inp_name, graph_name, output_name):
  inp = np.load(inp_name, allow_pickle=True)
  inp = np.reshape(inp, (1,224,224,3))
  g = load_pb(graph_name)
  i = g.get_operation_by_name("Placeholder")
  with g.as_default():
    with tf.Session() as sess:
      out = sess.run(g.get_operation_by_name(output_name).outputs[0], feed_dict={i.outputs[0]:inp})
      np.save("tensorflow_output", out)
      print("\n"*5)
      print("Output dumped in tensorflow_output.npy")
      print("Output = ", out)

if __name__ == '__main__':
  run(sys.argv[1], "model.pb", "fc/fc")