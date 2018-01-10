# ASQ-Solvers
Library of baseline solvers for Aristo Science Questions (ASQ) Set (TBD URL). These solvers 
retrieve relevant sentences from a large text corpus (available at TBD), use an entailment model 
to compute the score for each `(retrieved sentence, question+answer choice as an assertion)` pair
 and return the answer choices with the highest support i.e. entailment score. 
 
 ## Setup environment
 1. Create the `asq_solvers` environment using Anaconda
 
   ```
   conda create -n asq_solvers python=3.6
   ```
 
 2. Activate the environment
 
   ```
   source activate asq_solvers
   ```
 
 3. Install the requirements in the environment: 
 
   ```
   sh scripts/install_requirements.sh
   ```
 
 4. Install pytorch as per instructions on <http://pytorch.org/>. Command as of Jan. 9, 2018:
 
   ```
   conda install pytorch torchvision -c soumith`
   ```
  

 ## Setup data/models
 1. Download the data and models into `data/` folder
  ```
  sh scripts/download_data.sh
  ```
  
 2. Index text corpus and run ElasticSearch
  ```
  TBD
  ``` 
 
 ## Running baseline models
 Run the DGEM solver against the challenge set i.e., `data/ASQ-Challenge/ASQ-Challenge-Test.jsonl`
  
   ```
    sh scripts/evaluate_solver.sh data/ASQ-Challenge/ASQ-Challenge-Test.jsonl data/models/dgem/
   ```
 
 ## Running against a new question set
 To run the baseline solvers against a new question set, create a file using the JSONL format.
 For example:
 ```
 {
    "id":"Mercury_SC_415702",
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
				  {"text":"dry palms","label":"A"},
				  {"text":"wet palms","label":"B"},
				  {"text":"palms covered with oil","label":"C"},
				  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "answerKey":"A"}
 ``` 
  Run the evaluation script on this new file using: `sh scripts/evaluate_solver.sh new_file.jsonl`
  
  
 ## Running a new model
  To run a new entailment model (implemented using AllenNLP), you need to 
   1. Create a `Predictor` that converts the input JSON to an `Instance` expected by your 
   entailment model. See [DecompAttPredictor](asq_solvers/service/predictors/decompatt_qa_predictor.py)
   for an example.
     
   2. Add your custom predictor to the [predictor overrides](blob/basic_solver/asq_solvers/commands/__init__.py#L7)
   For example, if your new model is registered using `my_awesome_model` and the predictor is 
   registered using `my_awesome_predictor`, add `"my_awesome_model": "my_awesome_predictor"` to 
   the `predictor_overrides`.
   
   3. Run the `evaluate_solver.sh` script with your learned model in `my_awesome_model/model.tar.gz`
    ```
     sh scripts/evaluate_solver.sh \
        data/ASQ-Challenge/ASQ-Challenge-Test.jsonl \ 
        my_awesome_model/
    ``` 
     
