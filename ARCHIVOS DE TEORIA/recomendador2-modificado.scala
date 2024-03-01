import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

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



val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Build the recommendation model using ALS on the training data
val als = new ALS().setMaxIter(10).setRegParam(0.1).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)

// Evaluate the model by computing the RMSE on the test data
// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
model.setColdStartStrategy("drop")
val predictions = model.transform(test)




val evaluator = new RegressionEvaluator()
evaluator.setMetricName("rmse")
evaluator.setLabelCol("rating")
evaluator.setPredictionCol("prediction")
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
