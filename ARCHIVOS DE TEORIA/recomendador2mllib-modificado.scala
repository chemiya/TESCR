import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Carga y traducción de los datos
val data = sc.textFile("/home/usuario/tescr/reducido.txt")

val ratings = data.map(_.split('\t') match { case Array(user, item, rate,date) => 
  Rating(user.toInt, item.toInt, rate.toDouble)
}
)

println("Elementos convertidos:")
val primerosCincoElementos = ratings.take(5)
primerosCincoElementos.foreach(println)
println("\n")




// Construccion del recomendador usando ALS y las opciones habituales 
val rank = 2
val numIterations = 10
val lambda=0.01
val model = ALS.train(ratings, rank, numIterations, lambda)

model.save(sc, "./miFiltroColaborativo")
val mismoModelo = MatrixFactorizationModel.load(sc, "./miFiltroColaborativo")

model.predict(1,1)

// Evaluacion del modelo usando nuestra propia función para calcular mse vs a la propuesta de MLLib
// podemos realizar todas las predicciones para todos los usuarios
val usersProducts = ratings.map { case Rating(user, product, rate) =>
  (user, product)
}
println("Elementos convertidos:")
val primerosCincoElementosProducts = usersProducts.take(5)
primerosCincoElementosProducts.foreach(println)
println("\n")




val predictions =
  model.predict(usersProducts).map { case Rating(user, product, ratePred) =>
    ((user, product), ratePred)
  }

val ratesAndPreds = ratings.map { case Rating(user, product, rateReal) =>
  ((user, product), rateReal)
}.join(predictions)
println("Elementos convertidos:")
val primerosCincoElementosRates = ratesAndPreds.take(5)
primerosCincoElementosRates.foreach(println)
println("\n")



// ejemplo de cómo calcular el error cuadrático medio
val MSE = ratesAndPreds.map { case ((user, product), (ratePred, rateReal)) =>
  val err = (ratePred - rateReal)
  err * err
}.mean()
println("Mean Squared Error = " + MSE)

// metodo MLLIb para calcular el MSE. Necesitamos repetir mismos cálculos para
// los pares (rateReal y ratePred) para cada (user, product)

import org.apache.spark.mllib.evaluation.RegressionMetrics

//val usersProducts = ratings.map { case Rating(user, product, rate) =>
//  (user, product)
//}
//val predictions =
//  model.predict(usersProducts).map { case Rating(user, product, ratePred) =>
//    ((user, product), ratePred)
//  }
//val ratesAndPreds = ratings.map { case Rating(user, product, rateReal) =>
//  ((user, product), rateReal)
//}.join(predictions)

val predictedAndTrue = ratesAndPreds.map{
  case ((user, product), (real, predicted)) => 
  (real, predicted)
}
println("Elementos convertidos:")
val primerosCincoElementosPred = predictedAndTrue.take(5)
primerosCincoElementosPred.foreach(println)
println("\n")


val regressionMetrics = new RegressionMetrics(predictedAndTrue)
println("MSE: " + regressionMetrics.meanSquaredError)
println("RMSE: " + regressionMetrics.rootMeanSquaredError)
