
## Installation


```bash
conda create --prefix /path/to/envs/sw python=3.8 pip
conda activate /path/to/envs/sw
pip install scienceworld==1.1.3
pip install -r requirements.txt
conda install -c conda-forge openjdk 

```





## Data prepare 

```bash
cd  fast_slow_agent/data_utils
python data_convert.py
```



## Train Fast Model 


```
cd fast_agent
bash ds_train.sh  
```


 
## ReAct baseline

```bash 
bash run_eval_react.sh
```

## reflecxion baseline

```bash 
bash run_eval_reflexion.sh
```

## SayCan baseline

```bash 
bash run_eval_saycan.sh
```

## SwiftSage 

Note that we name SwiftSage as `fast_slow_agent` in the codebase. 

```bash 
bash run_eval_fast_slow.sh
```

The logs will be saved and the scripts for showing results and doing analysis are in the `analysis` folder.