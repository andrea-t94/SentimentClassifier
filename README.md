TODO LIST

OVERALL
- put todos in tickets

ON MODEL TRAINING
- use on EC2 with pulled images
- test GPU on EC2
- develop script that send them back to my laptop
- create a tech debt/improvents section and add proposal of using S3 (both as mount to use datasets and for uploading trained models)
- specify how model training section with docker works in my bot architecture
- Train on cloud 1st model
- Push best 1st to HFHub and train 2nd model
- Script for saving model from EC2 machine back to mu laptop
- OPTIONAL FROM NOW ON
- Create central GitHub repo all models (both 1st and 2nd step) (as did be them https://github.com/cardiffnlp/timelms)
- small script for pushing selected best model to HFHub (and make training script save only models locally)
- Add on HF eval metrics, reference to GitHub, hyperparam and any tag to consider each card as a different model version (e.g. "no processing")
- test against TweetBert and this model https://huggingface.co/blog/sentiment-analysis-twitter
- If TweetBert is better, wrap it around torch to be served
- Fine-tune chosen model hps (like batch, max len) on new Twitter data
- Test lr+lr scheduler, different batch size (https://medium.com/distributed-computing-with-ray/
hyperparameter-optimization-for-transformers-a-guide-c4e32c6c989b)
- swith to W&B
- Logs idea: how many inputs are impacted by MAX_LEN in prod? Useful since hp tuned by training only
- Validation: against fear-greed index
- Analysis: Test if cleaning processing (copy from article) improve performance

ON MODE INFERENCE (SENTIMENT ANALYZER)
- Build API (how to load the model/tokenizer in Pytorch (.tar with state dict https://pytorch.org/tutorials/beginner/saving_loading_models.html))
NEXT STEPS
- Add Gantry and Grafana for prod logs
- Add logs for inference ML in prod (train-serving skew, data drift, uptime, latency, thorughtput) 
