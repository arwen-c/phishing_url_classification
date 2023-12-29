# ML

### Install RAPIDS

```shell
conda create --solver=libmamba -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia cudf=23.12 cuml=23.12 python=3.10 cuda-version=12.0
conda activate rapids-23.12
conda install tldextract tqdm 
```