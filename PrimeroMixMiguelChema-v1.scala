/* Master en Ingeniería Informática - Universidad de Valladolid
*
*  TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
*  Mini-Proyecto 1 : Recomendadores
*
*  Dataset: CDs and Vinyl
* 
*  Primer script
*
*  Grupo 2: Sergio Agudelo Bernal
*           Miguel Ángel Collado Alonso
*           José María Lozano Olmedo.
*/


// iniciar sesión como $> spark-shell --driver-memory 4g

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator,CrossValidatorModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import java.util.Calendar
import java.sql.Timestamp
import java.text.SimpleDateFormat


val PATH="/home/usuario/Regresion/"
val ARCHIVO="CDs_and_Vinyl-10k.csv"

// formato del archivo de datos: item,user,rating,timestamp separados por ,

case class Rating( itemId: String, userId: String, rating: Float, timestamp: Long)
def parseRating(str: String): Rating = {
  val fields = str.split(",")
  assert(fields.size == 4)
  Rating(fields(0).toString, fields(1).toString, fields(2).toFloat, fields(3).toLong)
}

val ratings = spark.read.textFile(PATH + ARCHIVO).map(parseRating).toDF()

// Eliminamos nulos
  
var dfCDsVinyl = ratings.na.drop()

// Exploración de los datos
   
println("Número total de registros: " + dfCDsVinyl.count())
println("Primeros 5 registros:")
dfCDsVinyl.show(5)
println("Estructura del DataFrame:")
dfCDsVinyl.printSchema()

println("Resumen estadístico de Rating:")
dfCDsVinyl.describe("rating").show()

println("Número de votos por valor:")
dfCDsVinyl.groupBy("rating").count().orderBy(desc("count")).withColumnRenamed("count", "cuenta").show()

val MinMaxTime = dfCDsVinyl.agg(min("timestamp"),max("timestamp")).head()

// Convertir el timestamp a formato de fecha
val dateFormat = new SimpleDateFormat("dd-MM-yyyy hh:mm")
val minFecha = new Timestamp(MinMaxTime.getLong(0) * 1000L) // Multiplicamos por 1000L porque Timestamp espera milisegundos
val maxFecha = new Timestamp(MinMaxTime.getLong(1) * 1000L) // Multiplicamos por 1000L porque Timestamp espera milisegundos
val minFechaStr = dateFormat.format(minFecha)
val maxFechaStr = dateFormat.format(maxFecha)

println("Mínimo valor de timestamp:" + MinMaxTime(0) + " -> " +  minFechaStr)
println("Máximo valor de timestamp:" + MinMaxTime(1) + " -> " +  maxFechaStr)

println("Número de productos (Item): " + dfCDsVinyl.select("itemId").distinct.count())
println("Número de usuarios (User): " + dfCDsVinyl.select("userId").distinct.count())




dfCDsVinyl.show()











val columnasCategoricas = Seq("itemId","userId")  


val indexadores = columnasCategoricas.map { columna =>
  new StringIndexer()
    .setInputCol(columna)
    .setOutputCol(s"${columna}Index")
}


indexadores.foreach { indexador =>
  dfCDsVinyl = indexador.fit(dfCDsVinyl).transform(dfCDsVinyl)
}





var dfCDsVinylCorregido = dfCDsVinyl.select($"itemIdIndex".alias("itemId"), $"userIdIndex".alias("userId"), $"rating".alias("rating"), $"timestamp".alias("timestamp"))
dfCDsVinylCorregido.show()







/*
//algoritmo eliminar opiniones falsas y sesgadas-------------------


println("Comienza comprobacion de los datos de los usuarios")
val valoresUnicosUsuario = dfCDsVinylCorregido.select("userId").distinct().collect()
val usuariosDiferentes=valoresUnicosUsuario.length
var contador=0

//recorro diferentes usuarios
for (fila <- valoresUnicosUsuario) {
  println(s"Evaluando datos de: $contador / $usuariosDiferentes")
  contador=contador+1
  val valorUnicoDouble = fila.getDouble(0)
  
  //filtro opiniones de cada usuario
  val dfFiltrado = dfCDsVinylCorregido.filter(col("userId") === valorUnicoDouble)
  //cuantas opiniones han hecho
  val filas=dfFiltrado.count()
  //desviacion estandar de los timestamp de sus opiniones
  val desviacionEstandar = dfFiltrado.agg(stddev("timestamp").alias("desviacion_estandar")).collect()(0).getAs[Double]("desviacion_estandar")
  
  
  //condicional para eliminar datos si cumple condiciones
  if(desviacionEstandar<31622400 && filas>50){
    dfCDsVinylCorregido= dfCDsVinylCorregido.filter(col("userId") =!= valorUnicoDouble)
    val numeroFilas = dfCDsVinylCorregido.count()
    println(s"Se han eliminado los registros del usuario por no cumplir condicion 1. El número de filas en el DataFrame es: $numeroFilas")
  }


  //cuantas opiniones con cada rating han hecho
  var cuentaOpiniones=dfFiltrado.groupBy("rating").count().orderBy(desc("count")).withColumnRenamed("count", "cuenta")
  //convertimos en porcentaje
  cuentaOpiniones=cuentaOpiniones.select(format_number((col("cuenta") / filas)*100, 2).alias("Resultado"))
  //porcentaje opiniones rating 5
  val primerValorPorcentaje = cuentaOpiniones.select("Resultado").first().getAs[String](0)
  val valorDouble = primerValorPorcentaje.toDouble
  
  //condicional para eliminar datos si cumple condiciones
  if(valorDouble>90 && filas>50 ){
    dfCDsVinylCorregido= dfCDsVinylCorregido.filter(col("userId") =!= valorUnicoDouble)
    val numeroFilas = dfCDsVinylCorregido.count()
    println(s"Se han eliminado los registros del usuario por no cumplir condicion 2. El número de filas en el DataFrame es: $numeroFilas")
  }

}


*/

















// Particionado del Dataset



// **** particionado real ****
 val Array(training, test) = dfCDsVinylCorregido.randomSplit(Array(0.8, 0.2))


test.write.mode("overwrite").csv(PATH+"conjunto-test")
println("Conjunto de test guardado")


val als = new ALS().setUserCol("userId").setItemCol("itemId").setRatingCol("rating")

//val paramGrid = new ParamGridBuilder().addGrid(als.rank, Array(2,3,5)).addGrid(als.regParam, Array(0.01, 0.1,0.2)).addGrid(als.maxIter, Array(5,7,10)).build()
val paramGrid = new ParamGridBuilder().addGrid(als.rank, Array(2,3)).addGrid(als.regParam, Array(0.01, 0.1)).addGrid(als.maxIter, Array(5,7)).addGrid(als.alpha, Array(0.01)).build()


val evaluator = new RegressionEvaluator()
evaluator.setMetricName("rmse")
evaluator.setLabelCol("rating")
evaluator.setPredictionCol("prediction")


val cv1 = new CrossValidator().setEstimator(als).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(2)

println(s"Inicio: ${Calendar.getInstance.getTime}")


val cvmodel1 = cv1.fit(training)

println(s"Fin: ${Calendar.getInstance.getTime}")

val model = cvmodel1.bestModel.asInstanceOf[ALSModel]

//println(s"Mejor valor para 'rank': ${model.rank}")
//println(s"Mejor valor para 'maxIter': ${model.getOrDefault(model.maxIter)}")
//println(s"Mejor valor para 'regParam': ${model.getRegParam}")
//println(s"Mejor valor para 'alpha': ${model.getAlpha}")

model.write.overwrite().save(PATH+"modeloALS")


model.setColdStartStrategy("drop")
val predictions = model.transform(test)

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
