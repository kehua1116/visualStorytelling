
## Visual Storytelling with Emotions
In this project we explored the visual storytelling task with guided emotions. We optionally combined the common-sense concept network from [Chen et.al](https://arxiv.org/pdf/2102.02963.pdf)  and emotion extraction network to generate stories with more emotional expressiveness. Our trained emotion models are able to achieve better performance than the baseline model in several human-evaluated metrics. 

### Data Generation
- VIST dataset: requested from http://visionandlanguage.net/VIST/dataset.html
	- Combined images & text info for all stories: (not uploaded)
		 `data/vist/data/visualstorytelling/train.pkl`
		 `data/vist/data/visualstorytelling/test.pkl`
- Generate image features from ResNet152: extract from http://nlp.cs.ucsb.edu/data/VIST_resnet_features.zip
	- Results saved at: `data/vist/data/AREL/dataset/resnet_features/fc` (not uploaded)
- Generate common-sense concept keywords for each image (from acknowledgement): 
	- Code: `./Common_sense/concept_selection/train.py`
	- Results saved at: `data/vist/data/clarifai/train/` (not uploaded)
- Generate emotion keywords for each image:
	- Code: `./Emotion_detection/emotion_detection.ipynb`
	- Results saved at: `data/vist/emotion_annotations` (not uploaded)

### Model Training
(large folder: `./Common_sense/visualstorytelling`)
- Configurations: `opts.py`
- Model Architectures: `bart.py`, `bart_utils.py`
- Training Script: `train.py`, `dataset.py`
	- Example Commands:
		- baseline: `python3 train.py --with_concepts False --emotion "no_emotion"`
		- SingleE: `python3 train.py --with_concepts False --emotion "single_emotion" --use_synonyms True`
		- MultiE + Concept: `python3 train.py --with_concepts True--emotion "multi_emotion" --use_synonyms True`
	- Models saved at:  https://drive.google.com/drive/folders/1HY0o8229PLQn2Ex76DLrELt7Z8cQlftE?usp=share_link
		- SingleE:`4_single_emotion_noconcept.pt`
		- SingleE + Concept: `4_single_emotion_concept.pt`
		- MultiE: `4_multi_emotion_noconcept.pt`
		- MultiE + Concept: `4_multi_emotion_concept_0.9.pt`

### Model Evaluations
(large folder: `./Common_sense/visualstorytelling`)
- Evaluation Script:`evaluation.py`, `generator.py`
	- Example command: 
		`python3 evaluation.py --with_concepts False --emotion "no_emotion"`
	- Evaluation results saved at: `res`
- Evaluation Metrics: `vist_eval`
	- newly added `/bert_score`, others attributed on acknowledgements
- Human Evaluations:
	- Annotation generation & Results evaluation: `./Emotion_detection/human_evaluation.ipynb`
	- Annotation forms and results: https://drive.google.com/drive/folders/1elGnH582G82ws_FKfZAnWCN1Ez2xoNpv?usp=sharing


### Acknowledgements
- [VIST Evaluation Code](https://github.com/lichengunc/vist_eval)
- [Common-sense Concepts](https://github.com/sairin1202/Commonsense-Knowledge-Aware-Concept-Selection-For-Diverse-and-Informative-Visual-Storytelling)
- [VIST visualization API](https://github.com/lichengunc/vist_api)

