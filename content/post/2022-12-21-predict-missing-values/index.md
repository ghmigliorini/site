---
title: {{< staticref "/content/post/2022-12-21-predict-missing-values/index.ipynb" "newtab" >}}Python - Predicting missing values{{< /staticref >}}
subtitle: Using machine learning algorithms to predict missing values
summary: Using machine learning algorithms to predict missing values
authors:
- admin
tags: []
categories: []
date: "2023-02-13T00:00:00Z"
lastMod: "2023-02-13T00:00:00Z"
featured: false
draft: false
image:
  caption:
  focal_point: ""
---

```python
# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from palmerpenguins import load_penguins
```

```python
penguins = load_penguins()
penguins.head()
```

```python
penguins.shape
```

```python
print(penguins.isnull().sum())
```

```python
penguins_shuffle=shuffle(penguins, random_state=42)
```

```python

```

```python

```

