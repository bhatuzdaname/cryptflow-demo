import tensorflow as tf
import numpy as np

import sys
from run_tf import run
from pre_process import pre_process

if __name__ == '__main__':
    img_name = sys.argv[1]
    output_name = "xray"
    pre_process(img_name, output_name)
    run(output_name + ".npy", "model.pb", "fc/fc")
