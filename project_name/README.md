## Data installation

To download the data you need to have a Kaggle account to use the API. This can easily bedone by the following steps:

```pip install kaggle```

This can optionally be done in a virtual environment:


```python3 -m venv kaggle-env```

```source kaggle-env/bin/activate```


Either after or before you installed the kaggle API, you need to generate a token and put it the root directory:

- Go to your Kaggle settings -> account -> Create New Token

- This token will be downloaden and put this token in your root directory.

- ```mkdir -p ~/.kaggle```

- ```mv ~/Downloads/kaggle.json ~/.kaggle/```

- ```chmod 600 ~/.kaggle/kaggle.json``` for the right permissions


