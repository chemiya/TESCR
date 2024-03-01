import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator,CrossValidatorModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder

import java.util.Calendar

val PATH = "/home/usuario/tescr/"
val FILE = "reducido.txt"  

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
def parseRating(str: String): Rating = {
  val fields = str.split("\t")
  assert(fields.size == 4)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}

val ratings = spark.read.textFile(PATH + FILE).map(parseRating).toDF()
println("Elementos convertidos:")
ratings.show(5)
println("\n")
val Array(train, test) = ratings.randomSplit(Array(0.8, 0.2))




// Build the recommendation model using ALS on the training data
//val als = new ALS().setMaxIter(15).setRegParam(0.1).setNonnegative(true).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val als = new ALS().setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

val paramGrid = new ParamGridBuilder().addGrid(als.rank, Array(2,3,5)).addGrid(als.regParam, Array(0.01, 0.1,0.2)).addGrid(als.maxIter, Array(5,7,10)).build()

val evaluator = new RegressionEvaluator()
evaluator.setMetricName("rmse")
evaluator.setLabelCol("rating")
evaluator.setPredictionCol("prediction")




// Definimos el CrossValidator que realizará las pruebas
val cv1 = new CrossValidator().setEstimator(als).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(2)

println(s"Inicio: ${Calendar.getInstance.getTime}")

// Lanzamos el proceso
val cvmodel1 = cv1.fit(train)

println(s"Fin: ${Calendar.getInstance.getTime}")

val model = cvmodel1.bestModel.asInstanceOf[ALSModel]

model.getRank






// Evaluate the model by computing the RMSE on the test data
// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")





// Generate top 10 movie recommendations for each user
val userRecs = model.recommendForAllUsers(10)
println(userRecs.getClass)
userRecs.show()
// Generate top 10 user recommendations for each movie
val movieRecs = model.recommendForAllItems(10)
println(movieRecs.getClass)
movieRecs.show()

// Generate top 10 movie recommendations for a specified set of users
val users = ratings.select(als.getUserCol).distinct().limit(10)
println(users.getClass)
users.show()
val userSubsetRecs = model.recommendForUserSubset(users, 10)
println(userSubsetRecs.getClass)
userSubsetRecs.show()
// Generate top 10 user recommendations for a specified set of movies
val movies = ratings.select(als.getItemCol).distinct().limit(10)
println(movies.getClass)
movies.show()
val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
println(movieSubSetRecs.getClass)
movieSubSetRecs.show()
