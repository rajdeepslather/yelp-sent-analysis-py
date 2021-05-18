import pyspark.sql.functions as F
import sparknlp
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from sparknlp.annotator import UniversalSentenceEncoder, SentimentDLModel
from sparknlp.base import DocumentAssembler, Pipeline

spark = sparknlp.start()


def sent(r):
    meta = r[0][4]
    return float(meta['positive']) - float(meta['negative'])


# custom pipeline with pre-trained models
document = DocumentAssembler().setInputCol("text").setOutputCol("document")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en").setInputCols(["document"]).setOutputCol(
    "sentence_embeddings")

# other pre-trained sentiment classification model: sentimentdl_use_twitter
# the classes/labels/categories are in category column
sentimentdl = SentimentDLModel.pretrained(name="sentimentdl_use_imdb", lang="en").setInputCols(
    ["sentence_embeddings"]).setOutputCol("sentiment")
pipeline = Pipeline(
    stages=[
        document,
        use,
        sentimentdl
    ])
pipeline_model = pipeline.fit([])  # already trained

# yelp_review = spark.read.json(path+'/review.json')
yelp_review = spark.read.json('data/yelp_academic_dataset_review.json')
yelp_review = yelp_review.drop('date')

# need to get sentiment for all data (train and test)
# yelp_review_annotated = pipeline.annotate(yelp_review, 'text')
yelp_review_annotated = pipeline_model.transform(yelp_review)
yelp_review.unpersist()

yelp_review_annotated = yelp_review_annotated.withColumn('stars_thresh', yelp_review_annotated.stars - 2.5)
yelp_review_annotated = yelp_review_annotated.withColumn('stars_thresh_abs', F.abs(yelp_review_annotated.stars_thresh))
yelp_review_annotated = yelp_review_annotated \
    .withColumn('stars_sent', yelp_review_annotated.stars_thresh / yelp_review_annotated.stars_thresh_abs) \
    .drop('stars_thresh_abs', 'stars_thresh')

sent_udf = F.udf(sent)

yelp_review_annotated = yelp_review_annotated.withColumn('sent_to_num', sent_udf(yelp_review_annotated.sentiment))
yelp_review_annotated = yelp_review_annotated.withColumn('sent_to_num',
                                                         yelp_review_annotated.sent_to_num.cast(DoubleType()))

# yelp_user = spark.read.json(path+'/user.json')
yelp_user = spark.read.json('data/yelp_academic_dataset_user.json')

yelp_user = yelp_user.select('user_Id', 'review_count', 'average_stars', 'useful', 'fans')

# yelp_user
yelp_review_sents = yelp_review_annotated.select('stars', 'user_Id', 'sent_to_num', 'useful', 'funny', 'cool')
yelp_review_annotated.unpersist()

# yelp_review_sents
reviews_j = yelp_review_sents.join(yelp_user, yelp_review_sents.user_Id == yelp_user.user_Id)

yelp_user.unpersist()
yelp_review_sents.unpersist()

reviews_j = reviews_j.drop('user_Id')  # we don't need user_Id after this
reviews_j = reviews_j.toDF('stars', 'sent_to_num', 'useful_review', 'funny', 'cool', 'user_review_count',
                           'user_average_stars', 'user_useful', 'user_fans')

# assemble all features together into one vector
assembler = VectorAssembler().setInputCols(
    ['sent_to_num', 'useful_review', 'funny', 'cool', 'user_review_count', 'user_average_stars', 'user_useful',
     'user_fans']).setOutputCol("features")

# transform train, test sets: (stars, [f1, f2, f3, ..., fn])
inputDF = assembler.transform(reviews_j).select("stars", "features")

reviews_j.unpersist()
# split into train and test sets
train, test = inputDF.randomSplit(weights=[0.75, 0.25], seed=1)

# split train into development and validation
development, validation = train.randomSplit(weights=[0.75, 0.25], seed=1)

# use development set to train network, validation set to test network to obtain optimal parameters
# retrain network on train set with optimal parameters
# test network on test set


# development, validation sets used to find optimal parameters
# once optimal parameters are found, retrain network on train set

# specify layers for the neural network:
# input layer of size 8 (features), one intermediate of size 12
# and output of size 6 (classes)
layers = [8, 12, 6]
trainer = MultilayerPerceptronClassifier(maxIter=40, layers=layers,
                                         blockSize=128, seed=1234,
                                         labelCol='stars', stepSize=0.1)

# train the model with development set
model = trainer.fit(development)

# compute accuracy on the validation set
result = model.transform(validation)

predictionAndLabels = result.select("prediction", "stars")
# result.unpersist()

evlauator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="stars", metricName="rmse")
rmse = evlauator_rmse.evaluate(predictionAndLabels)

print("RMSE on Validation Set: " + str(rmse))

# I'm using only RMSE for validation
evlauator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="stars", metricName="mae")
mae = evlauator_mae.evaluate(predictionAndLabels)

print("MAE on Validation Set: " + str(mae))

# maxIter=40, layers=[8, 12, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.4109812827756614  MAE on Validation Set: 0.814901367388652
# ^ Last Exp with maxIter 

# maxIter=80, layers=[8, 12, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.41099998082853 MAE on Validation Set: 0.815022064679219

# maxIter=80, layers=[8, 6, 4, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.5410209010637315 MAE on Validation Set: 0.9124236252545825

# maxIter=80, layers=[8, 6, 12, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.5674471918547126 MAE on Validation Set: 0.9429735234215886

# maxIter=40, layers=[8, 18, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.420025700678034 MAE on Validation Set: 0.815443627144308
# ^ Exp with layers (increased maxIter cause of increased complexity)

# maxIter=40, layers=[8, 6, 6], stepSize=0.2, solver='l-bfgs'
# RMSE on Validation Set: 1.53800977406008 MAE on Validation Set: 0.918837407325199

# maxIter=40, layers=[8, 6, 6], stepSize=0.01, solver='l-bfgs'
# RMSE on Validation Set: 1.4889757553339852 MAE on Validation Set: 0.8795244799782438
# ^ Exp with stepSize

# maxIter=40, layers=[8, 6, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.4886020026926277 MAE on Validation Set: 0.8787903477908067
# ^ Exp with maxIter

# maxIter=80, layers=[8, 6, 6], stepSize=0.1, solver='l-bfgs'
# RMSE on Validation Set: 1.4992918469644249 MAE on Validation Set: 0.8730931519305250

# maxIter=200, layers=[8, 6, 6], stepSize=0.1, solver='gd'
# RMSE on Validation Set: 1.8845096442689286 MAE on Validation Set: 1.1528520453096466

# maxIter=80, layers=[8, 6, 6], stepSize=0.1, solver='gd'
# RMSE on Validation Set: 1.955525563079031 MAE on Validation Set: 1.2077877931603328
# ^ Exp with solver

# retrain network with optimal parameters on the train set
# test network on test set
layers = [8, 12, 6]
trainer = MultilayerPerceptronClassifier(maxIter=40, layers=layers,
                                         blockSize=128, seed=1234,
                                         labelCol='stars', stepSize=0.1)

# train the model with full train set
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)

predictionAndLabels = result.select("prediction", "stars")
result.unpersist()

evlauator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="stars", metricName="rmse")
rmse = evlauator_rmse.evaluate(predictionAndLabels)

print("RMSE on Test Set: " + str(rmse))

# I'm using only RMSE for validation
evlauator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="stars", metricName="mae")
mae = evlauator_mae.evaluate(predictionAndLabels)

print("MAE on Test Set: " + str(mae))
