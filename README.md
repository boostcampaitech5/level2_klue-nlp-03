# level2-klue-nlp-03

## 🌱Members

|<img src='https://avatars.githubusercontent.com/u/110003154?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60145579?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/54995090?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/75467530?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/65614582?v=4' height=100 width=100px></img>|
| --- | --- | --- | --- | --- |
| [김민혁](https://github.com/torchtorchkimtorch) | [김의진](https://github.com/KimuGenie) | [김성우](https://github.com/tjddn0402) | [오원택](https://github.com/dnjdsxor21) | [정세연](https://github.com/jjsyeon) |

## Relation Extraction 
Relation Extraction is the task of predicting attributes and relations between subject and object entity in sentence. 
![image](https://www.mdpi.com/2079-9292/9/10/1637)

## Getting started

```python
# Install Packages
pip install -r requirements.txt

# Train & Inference
python src/train.py
```
  
### 📂Structure

```python
root/
|
|-- config.yaml
|-- config_sweep.yaml
|
|-- src/
|   |-- train.py
|   |-- train_sweep.py
|   |-- loader.py
|   |-- models.py
|   |-- utils.py
|
|-- eda/
|   |-- entity_EDA.ipynb
|   |-- result_eda.ipynb
```

## Data
#### Augmentation
- [Easy Data Augmentation](https://github.com/toriving/KoEDA)
- [Back Translation](https://github.com/ssut/py-googletrans)
#### Preprocessing
- [Hanja Removal](https://github.com/suminb/hanja)
- [Unidecode](https://github.com/avian2/unidecode)

## Model
- [Entity Marker](https://arxiv.org/abs/2102.01373)
- [Binary Classifier](https://www.kaggle.com/code/duongthanhhung/bert-relation-extraction)
- [RECENT(entity relation restriction)](https://arxiv.org/abs/2105.08393)
- [Additional Model for Fine-tuning](https://arxiv.org/abs/1906.03158)
