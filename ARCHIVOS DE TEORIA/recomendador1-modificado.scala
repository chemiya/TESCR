import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

val PATH = "/home/usuario/tescr/"
val FILE = "reducido.txt"  

// case class para poder mover Ratings y usar los nombres de sus campos
case class Rating(userId: Int, itemId: Int, rating: Float)

// a partir de un string genera un objeto Rating()

def parseRating(str: String): Rating = {
  val fields = str.split("\t")
  assert(fields.size == 4)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
}


// leemos los datos y los convertimos en un DataFrame
val ratings = spark.read.textFile(PATH + FILE).map(parseRating).toDF()
println("Elementos convertidos:")
ratings.show(5)
println("\n")

// creamos una partición de los datos, training y test
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Construimos el modelo de recomendador usando ALS para el conjunto de training 
val als = new ALS().setMaxIter(10).setRegParam(0.1).setNonnegative(true).setUserCol("userId").setItemCol("itemId").setRatingCol("rating")
val model = als.fit(training)

// Evaluamos la bondad del modelo usando RMSE sobre las predicciones sobre el conjunto de test
// Cuando se usa la opción 'drop' en setColdStrategy nos aseguramos que no se usarán los valores NaN en el cálculo del error
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")

val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

// Obtenemos 3 recomendaciones para cada usuario 
val userRecs = model.recommendForAllUsers(3)
println(userRecs.getClass)
userRecs.show()
// Seleccionamos 3 recomendaciones de usuarios para cada item 
val itemRecs = model.recommendForAllItems(3)
println(itemRecs.getClass)
itemRecs.show()

// Generamos tres recomendaciones de items para un conjunto de usuarios 
val users = ratings.select(als.getUserCol).distinct().limit(3)
println(users.getClass)
users.show()
val userSubsetRecs = model.recommendForUserSubset(users, 3)
println(userSubsetRecs.getClass)
userSubsetRecs.show()





// Generamos 3 recomedaciones para un conjunto de items
val items = ratings.select(als.getItemCol).distinct().limit(3)
println(items.getClass)
items.show()
val itemSubSetRecs = model.recommendForItemSubset(items, 3)
println(itemSubSetRecs.getClass)
itemSubSetRecs.show()