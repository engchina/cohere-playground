# langchain-cohere
Integration of Langchain and Cohere

## prepare

create conda environment,

```
conda create -n langchain-cohere python=3.10 -y
conda activate langchain-cohere
```

install requirements,

```
pip install -r requirements.txt
```

config cohere api key 

```
cp .env.example .env
```

## launch

```
python main.py
```

## access 

open [http://127.0.0.1:7860/?__theme=dark](http://127.0.0.1:7860/?__theme=dark)