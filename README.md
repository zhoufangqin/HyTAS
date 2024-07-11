# HyTAS: A Hyperspectral Image Transformer Architecture Search Benchmark and Analysis (ECCV 2024)


## Pipeline
![Image](diagram.pdf)


### To run the search scripts, specify a search proxy in the search_xx.sh, then run with bash
### As an exmple below:
```bash search_indian_sf_final.sh
```
### Because the datasets are quite large, we only included Indian Pines in the dataset folder.

### To run the retrain scripts, specifiy a dataset and path in the train_xx.sh, then run with bash
```bash train_searched_sf.sh
```

### analysis.ipynb is the notebook for visualizing the results

### Individual retrain and search results are saved under outputs/ 
### Merged results of each dataset are saved under results/ 


