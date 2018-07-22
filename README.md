# ARC-Solvers
Library of baseline solvers for AI2 Reasoning Challenge (ARC) Set (http://data.allenai.org/arc/).
These solvers retrieve relevant sentences from a large text corpus (ARC_Corpus.txt in the
dataset), and use two types of models to predict the correct answer.
 1. An entailment-based model that computes the entailment score for each `(retrieved sentence,
 question+answer choice as an assertion)` pair and scores each answer choice based on the
 highest-scoring sentence.
 2. A reading comprehension model (BiDAF) that converts the retrieved sentences into a paragraph
 per question. The model is used to predict the best answer span and each answer choice is scored
  based on the overlap with the predicted span.
 
 ## Setup environment
 1. Create the `arc_solvers` environment using Anaconda
 
   ```
   conda create -n arc_solvers python=3.6
   ```
 
 2. Activate the environment
 
   ```
   source activate arc_solvers
   ```
 
 3. Install the requirements in the environment: 
 
   ```
   sh scripts/install_requirements.sh
   ```
 
 4. Install pytorch as per instructions on <http://pytorch.org/>. Command as of Feb. 26, 2018:
 
   ```
   conda install pytorch torchvision -c pytorch
   ```
  

 ## Setup data/models
 1. Download the data and models into `data/` folder. This will also build the ElasticSearch
 index (assumes ElasticSearch 6+ is running on `ES_HOST` machine defined in the script)
  ```
  sh scripts/download_data.sh
  ```

 2. Download and prepare embeddings. This will download glove.840B.300d.zip from https://nlp.stanford.edu/projects/glove/ and 
 convert it to glove.840B.300d.txt.gz which is readible from AllenNLP
   ```
  sh download_and_prepare_glove.sh
  ```
  
   
 
 ## Running baseline models
 Run the entailment-based baseline solvers against a question set using `scripts/evaluate_solver.sh`

 For example, to evaluate the DGEM model on the Challenge Set, run:
```
sh scripts/evaluate_solver.sh \
	data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl \
	data/ARC-V1-Models-Feb2018/dgem/
```
  Change `dgem` to `decompatt` to test the Decomposable Attention model.

 To evaluate the BiDAF model, use the `evaluate_bidaf.sh` script
```
 sh scripts/evaluate_bidaf.sh \
    data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl \
    data/ARC-V1-Models-Feb2018/bidaf/
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
  Run the evaluation scripts on this new file using the same commands as above.
  
  
 ## Running a new Entailment-based model
  To run a new entailment model (implemented using AllenNLP), you need to 
   1. Create a `Predictor` that converts the input JSON to an `Instance` expected by your 
   entailment model. See [DecompAttPredictor](arc_solvers/service/predictors/decompatt_qa_predictor.py)
   for an example.
     
   2. Add your custom predictor to the [predictor overrides](arc_solvers/commands/__init__.py#L8)
   For example, if your new model is registered using `my_awesome_model` and the predictor is 
   registered using `my_awesome_predictor`, add `"my_awesome_model": "my_awesome_predictor"` to 
   the `predictor_overrides`.
   
   3. Run the `evaluate_solver.sh` script with your learned model in `my_awesome_model/model.tar.gz`:

    ```
     sh scripts/evaluate_solver.sh \
        data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl \
        my_awesome_model/
    ```
     
## Running a new Reading Comprehension model
 To run a new reading comprehension (RC) model (implemented using AllenNLP), you need to
   1. Create a `Predictor` that converts the input JSON to an `Instance` expected by your
   RC model. See [BidafQaPredictor](arc_solvers/service/predictors/bidaf_qa_predictor.py)
   for an example.

   2. Add your custom predictor to the [predictor overrides](arc_solvers/commands/__init__.py#L8)
   For example, if your new model is registered using `my_awesome_model` and the predictor is
   registered using `my_awesome_predictor`, add `"my_awesome_model": "my_awesome_predictor"` to
   the `predictor_overrides`.

   3. Run the `evaluate_bidaf.sh` script with your learned model in `my_awesome_model/model.tar.gz`:

    ```
     sh scripts/evaluate_solver.sh \
        data/ARC-V1-Feb2018/ARC-Challenge/ARC-Challenge-Test.jsonl \
        my_awesome_model/
    ``` 
    
    
## Training the BiLSTM max-out Question to Choices Max Attention
To train the model, you need to have the data and embeddings downloaded.

```bash
python arc_solvers/run.py train -s trained_models/qa_multi_question_to_choices/serialization/ arc_solvers/training_config/qa/multi_choice/reader_qa_multi_choice_max_att_ARC_Chellenge_full.json
```
