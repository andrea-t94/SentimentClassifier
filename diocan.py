import boto3
import mlflow
import mlflow.sklearn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# custom
from datasets.wiki.dataset import WikiVocabulary
from datasets.utils import data_process, batchify, get_batch

digits = datasets.load_digits()

# dataset
batch_size = 20
eval_batch_size = 10
device = 'cpu'

tokenizer = get_tokenizer(tokenizer='basic_english')
vocab = WikiVocabulary(tokenizer=tokenizer, save_path='datasets/wiki/artifacts').get_vocab()
train_iter, val_iter, test_iter = WikiText2()
# batches of shape [seq_len, batch_size]
train_data = batchify(data_process(train_iter, vocab, tokenizer), batch_size).to(device)
val_data = batchify(data_process(val_iter, vocab, tokenizer), eval_batch_size).to(device)
test_data = batchify(data_process(test_iter, vocab, tokenizer), eval_batch_size).to(device)

print(test_data)
print(data_process(test_iter, vocab, tokenizer))
for item in test_iter:
    print(type(item))


mlflow.set_tracking_uri('http://ec2-13-59-144-81.us-east-2.compute.amazonaws.com')
mlflow.set_experiment(experiment_name='test2')

# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))
#
# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)
#
# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )
#
# # Learn the digits on the train subset
# clf.fit(X_train, y_train)
#
# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)
# mlflow.log_metric('accuracy',accuracy_score(y_test, predicted))
# #mlflow.sklearn.log_model(clf, 'svm')
#
# features = "rooms, zipcode, median_price, school_rating, transport"
# with open("features.txt", 'w') as f:
#     f.write(features)
#
# # Log the artifact in a directory "features" under the root artifact_uri/features
# mlflow.log_artifact("features.txt", artifact_path="features")
#
#
