# cryptflow-demo

#### Step 1. Load environment
```
source path/to/EzPC/mpc_venv/bin/activate
```

#### Step 2. Compile graph.
```
python path/to/EzPC/Athos/CompileTFGraph.py --config config.json --role server
```
#### Step 3. Compile for client (we can skip if testing on the same machine)
```
python path/to/EzPC/Athos/CompileTFGraph.py --config config.json --role client
```
#### Step 4. Preprocess xray.
```
python pre_process.py covid19positive.jpeg
```
This dumps input image as a numpy array (```xray.npy```)

#### Step 5. Scale input image.
```
python path/to/EzPC/Athos/CompilerScripts/convert_np_to_fixedpt.py --inp xray.npy --config config.json
```
This dumps ```xray_fixedpt_scale_12.inp```.

#### Step 6. Run computation.
Server on one machine:
```
./model_SCI_OT.out r=1 p=12345 < model_input_weights_fixedpt_scale_12.inp
```
Client on another machine:
```
./model_SCI_OT.out r=2 ip=123.231.231.123 p=12345 < xray_fixedpt_scale_12.inp > output.txt
```
You can change ip to 127.0.0.1 if both run on the same machine. (Will take around ~mins to run with a peak memory usage of )

#### Step 7. Extract output as numpy array
```
python path/to/EzPC/CompilerScripts/get_output.py output.txt config.json
```
Dumps ```model_output.npy``` as a flattened numpy array(1-D).

#### Step 8. Verify Results
Run input graph with tensorflow.
```
python run_tf.py xray.npy
```
Dumps output in ```tensorflow_output.npy```. Compare with output generated in 2PC computation:
```
python path/to/EzPC/CompilerScripts/comparison_scripts/compare_np_arrs.py tensorflow_output.npy model_output.npy
```

