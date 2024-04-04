### Introduction
For this project, we'll fine-tuned BERT model to detect fake news (including using statement or justification to predict binary label or multi-label). 

### Developer Setup
```
# load and activate the academic-ml conda environment on SCC
module load miniconda
module load academic-ml/spring-2024
conda activate yliu_env

# Add the path to your source project directory to the python search path
# so that the local `import` commands will work.
export PYTHONPATH="/projectnb/ds598/projects/<userid>/<yourdir>:$PYTHONPATH"
```
### DataSet
The dataset is LIAR-PLUS,
a extended fact-checking and fake news detection dataset release in [Where is Your Evidence: Improving Fact-checking by Justification Modeling](https://aclanthology.org/W18-5513/) . 
It includes news statement, fact-checking labels and evidence justifications extracted automatically from the full-text verdict report written by journalists in Politifact. 

### Evaluation
Binary Accuracy or Multi-class Accuracy based on statement or justification input (or combined)
