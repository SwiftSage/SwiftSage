
## Installation


```bash
conda create -n sw python=3.8 pip
conda activate sw
pip install scienceworld==1.1.3
pip install -r requirements.txt
conda install -c conda-forge openjdk # if needed 
```





## Imitation learning 

You can skip this step by simply using our checkpoint here: https://huggingface.co/yuchenlin/swift_sw
It is based on Flan-t5-large (770m).
### Generating data for imitation learning (behavior cloning)

```bash
cd  fast_slow_agent/data_utils/
# unzip goldpaths-all.zip 
python data_convert.py 
```



### Train Swift Module 

```
cd fast_agent
bash ds_train.sh  
```


## The SwiftSage Agent

Note that we name SwiftSage as `fast_slow_agent` in the codebase. 

```bash 
bash run_eval_fast_slow.sh
```

The logs will be saved and the scripts for showing results and doing analysis are in the `analysis` folder.

## Evaluation  
### ReAct baseline

```bash 
bash run_eval_react.sh
```

### reflecxion baseline

```bash 
bash run_eval_reflexion.sh
```

### SayCan baseline

```bash 
bash run_eval_saycan.sh
```

### Other baseline methods

Check out: https://github.com/allenai/ScienceWorld