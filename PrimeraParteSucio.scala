
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator,CrossValidatorModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import java.util.Calendar




//Carga del fichero-----------------------------------------------------
val spark = SparkSession.builder.appName("TESCR").getOrCreate()
val rutaArchivoJson = "/home/usuario/tescr/dataset-reducido.json"
var datosJson = spark.read.json(rutaArchivoJson)
datosJson.printSchema()
datosJson.show()



/*
val stringIndexer = new StringIndexer()
  .setInputCol("reviewerID")
  .setOutputCol("reviewerIDIndex")


val datosIndexado = stringIndexer.fit(datosJson).transform(datosJson)
datosIndexado.show()*/

/*
val attributeColumns = datosJson.columns.toSeq.filter(_ != "income").toArray

val outputColumns = attributeColumns.map(_ + "-num").toArray

val siColumns= new StringIndexer().setInputCols(attributeColumns).setOutputCols(outputColumns).setStringOrderType("alphabetDesc")

val simColumns = siColumns.fit(datosJson)

val datosJsonNumeric = simColumns.transform(datosJson).drop(attributeColumns:_*)
*/


val columnasCategoricas = Seq("reviewerID","asin")  


val indexadores = columnasCategoricas.map { columna =>
  new StringIndexer()
    .setInputCol(columna)
    .setOutputCol(s"${columna}Index")
}


indexadores.foreach { indexador =>
  datosJson = indexador.fit(datosJson).transform(datosJson)
}

datosJson.show()


val filtrado=datosJson.select("reviewerIDIndex","asinIndex","overall")
filtrado.show()




val Array(training, test) = filtrado.randomSplit(Array(0.8, 0.2))
test.write.csv("/home/usuario/tescr/conjunto-test.csv")
println("Conjunto de test guardado")







val als = new ALS().setUserCol("reviewerIDIndex").setItemCol("asinIndex").setRatingCol("overall")

val paramGrid = new ParamGridBuilder().addGrid(als.rank, Array(2,3,5)).addGrid(als.regParam, Array(0.01, 0.1,0.2)).addGrid(als.maxIter, Array(5,7,10)).build()

val evaluator = new RegressionEvaluator()
evaluator.setMetricName("rmse")
evaluator.setLabelCol("overall")
evaluator.setPredictionCol("prediction")




val cv1 = new CrossValidator().setEstimator(als).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(2)

println(s"Inicio: ${Calendar.getInstance.getTime}")


val cvmodel1 = cv1.fit(training)

println(s"Fin: ${Calendar.getInstance.getTime}")

val model = cvmodel1.bestModel.asInstanceOf[ALSModel]

//model.getRank
println(model)
model.write.overwrite().save("/home/usuario/tescr/modelo")




model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")




spark.stop()
