import pyspark.sql.functions as F
import sparknlp
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType
from sparknlp.annotator import UniversalSentenceEncoder, SentimentDLModel
from sparknlp.base import DocumentAssembler, Pipeline

spark = sparknlp.start()

# custom pipeline with pre-trained models
document = DocumentAssembler().setInputCol("text").setOutputCol("document")
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en").setInputCols(["document"]).setOutputCol(
    "sentence_embeddings")

# other pre-trained sentiment classification model: sentimentdl_use_twitter
# the classes/labels/categories are in category column
# sentimentdl_use_twitter
sentimentdl = SentimentDLModel.pretrained(name="sentimentdl_use_imdb", lang="en").setInputCols(
    ["sentence_embeddings"]).setOutputCol("sentiment")
pipeline = Pipeline(
    stages=[
        document,
        use,
        sentimentdl
    ])
pipeline_model = pipeline.fit([])  # already trained

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


# This is for discrete results
def sent(r):
    return r[0][3]


def sent_to_num(r):
    if r == 'positive':
        return 1.0
    else:
        return -1.0


sent_udf = F.udf(sent)
sent_to_num_udf = F.udf(sent_to_num)

# This is for discrete results
yelp_review_annotated = yelp_review_annotated.withColumn('sent', sent_udf(yelp_review_annotated.sentiment))
yelp_review_annotated = yelp_review_annotated.withColumn('sent_to_num', sent_to_num_udf(yelp_review_annotated.sent))
yelp_review_annotated = yelp_review_annotated.withColumn('sent_to_num',
                                                         yelp_review_annotated.sent_to_num.cast(DoubleType()))

predictionAndLabels = yelp_review_annotated.select('sent_to_num', 'stars_sent')
evaluator = MulticlassClassificationEvaluator(metricName="accuracy",
                                              predictionCol="sent_to_num",
                                              labelCol="stars_sent")
print("accuracy = ", evaluator.evaluate(predictionAndLabels))
