/*
    Master en Ingeniería Informática - Universidad de Valladolid

    TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
    Mini-Proyecto 2 : Regresión

    Dataset: Washington Bike Sharing Dataset Data Set
    
    Grupo 2:
    Sergio Agudelo Bernal
    Miguel Ángel Collado Alonso
    José María Lozano Olmedo
*/

// iniciar sesión como $> spark-shell --driver-memory 4g

// Configuraciones iniciales
//sc.setCheckpointDir("checkpoint")
//sc.setLogLevel("ERROR")

// Importación de módulos a usar


import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.regression.LinearRegressionSummary
import org.apache.spark.ml.evaluation.RegressionEvaluator 

// ----------------------------------------------------------------------------------------
// Carga de los datos
// ----------------------------------------------------------------------------------------

println("\nCARGA DE LOS DATOS")

val PATH="/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO="hour.csv"


val bikeDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load(PATH+ARCHIVO)
println("Datos cargados:")
bikeDF.show(10)






// ----------------------------------------------------------------------------------------
// Exploración de los datos
// ----------------------------------------------------------------------------------------

println("\nEXPLORACIÓN DE LOS DATOS")

// Definición del esquema de datos

bikeDF.printSchema

// Comprobamos que no hay valores nulos

bikeDF.select(bikeDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

// Estadística de los valores de los atributos

// Atributos numéricos

bikeDF.describe("instant").show()
bikeDF.describe("casual").show()
bikeDF.describe("registered").show()
bikeDF.describe("cnt").show()
bikeDF.describe("temp").show()
bikeDF.describe("atemp").show()
bikeDF.describe("hum").show()
bikeDF.describe("windspeed").show()

// Atributos categóricos

println("Número de alquileres por época del año:")
bikeDF.
    groupBy("season").
    count().
    orderBy(asc("season")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por año:")
bikeDF.
    groupBy("yr").
    count().
    orderBy(asc("yr")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por mes:")
bikeDF.
    groupBy("mnth").
    count().
    orderBy(asc("mnth")).
    withColumnRenamed("count", "cuenta").
    show()
	
println("Número de alquileres por hora:")
bikeDF.
    groupBy("hr").
    count().
    orderBy(asc("hr")).
    withColumnRenamed("count", "cuenta").
    show(24)
	
println("Número de alquileres por día de la semana:")
bikeDF.
    groupBy("weekday").
    count().
    orderBy(asc("weekday")).
    withColumnRenamed("count", "cuenta").
    show()

println("Número de alquileres por meteorología:")
bikeDF.
    groupBy("weathersit").
    count().
    orderBy(asc("weathersit")).
    withColumnRenamed("count", "cuenta").
    show()	

println("Número de alquileres por día festivo:")
bikeDF.
    groupBy("holiday").
    count().
    orderBy(asc("holiday")).
    withColumnRenamed("count", "cuenta").
    show()	
	
println("Número de alquileres por día laboral:")
bikeDF.
    groupBy("workingday").
    count().
    orderBy(asc("workingday")).
    withColumnRenamed("count", "cuenta").
    show()	









//Eliminar atributos 

val columnasAEliminar = Seq("instant", "dteday", "atemp", "windspeed", "casual", "registered")
val nuevoDF = bikeDF.drop(columnasAEliminar: _*)
nuevoDF.show()






























// ----------------------------------------------------------------------------------------
// Transformación de los datos
// ----------------------------------------------------------------------------------------


/*
println("\nTRASNFORMACIÓN DE LOS DATOS")

// Transformación de los atributos categoricos con OneHotEncoder

val indexer = Array("season","yr","mnth","hr","weekday","weathersit").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
val pipeline = new Pipeline().setStages(indexer)
val bikeDF_trans = pipeline.fit(nuevoDF).transform(nuevoDF).drop("season","yr","mnth","hr","weekday","weathersit")
bikeDF_trans.show()
*/





















// ----------------------------------------------------------------------------------------
// Partición de los datos en train y test
// ----------------------------------------------------------------------------------------

println("\nPARTICIÓN DE LOS DATOS")

val splitSeed = 123
val Array(trainingData,testData) = nuevoDF.randomSplit(Array(0.7,0.3),splitSeed)

//Selección y ensamblado de columna feature

//val feature = Array("holiday","workingday","temp","hum","season_Vec","yr_Vec","mnth_Vec","hr_Vec","weekday_Vec","weathersit_Vec")
val feature = Array("holiday","workingday","temp","hum","season","yr","mnth","hr","weekday","weathersit")

val assembler = new VectorAssembler().setInputCols(feature).setOutputCol("features")

















// ----------------------------------------------------------------------------------------
// Selección de mejor modelo
// ----------------------------------------------------------------------------------------

println("\nSELECCIÓN DE MODELO")









































































/*
// ----------------------------------------------------------------------------------------
// Guardamos el mejor modelo (GBT Regressor)
// ----------------------------------------------------------------------------------------

println("\nGUARDAMOS EL MEJOR MODELO (GBT Regressor)")


//Guardamos el mejor modelo
gbtModel.write.overwrite().save(PATH+"/modelo")



*/




































//Validacion cruzada con evaluacion de parametros sobre linearRegression


// Definir el modelo de regresión lineal
val lr = new LinearRegression()
  .setLabelCol("cnt")
  .setFeaturesCol("features")

// Crear el pipeline
val pipeline = new Pipeline()
  .setStages(Array(assembler, lr))



// Definir el grid de parámetros
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()

// Definir el evaluador de regresión
val evaluator = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Definir el validador cruzado
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajustar el modelo utilizando el conjunto de entrenamiento
val cvModel = cv.fit(trainingData)

// Evaluar el modelo en el conjunto de prueba
val predictions = cvModel.transform(testData)
val rmse = evaluator.evaluate(predictions)




// Imprimir los parámetros del mejor modelo
val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val lrModel = bestModel.stages(1).asInstanceOf[LinearRegressionModel]
println(s"Best model parameters: regParam = ${lrModel.getRegParam}, elasticNetParam = ${lrModel.getElasticNetParam}")

val metricas = evaluator.getMetrics(predictions)
//Algunas métricas
println(s"MSE: ${metricas.meanSquaredError}")
println(s"r2: ${metricas.r2}")
println(s"root MSE: ${metricas.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas.meanAbsoluteError}")




























//Validacion cruzada con evaluacion de parametros sobre GBTRegressor
// Definir el modelo GBTRegressor
val gbt = new GBTRegressor()
  .setLabelCol("cnt")
  .setFeaturesCol("features")

// Crear el pipeline
val pipeline1 = new Pipeline()
  .setStages(Array(assembler, gbt))



// Definir el grid de parámetros
val paramGrid1 = new ParamGridBuilder().addGrid(gbt.maxDepth, Array(5, 10)).addGrid(gbt.maxIter, Array(10, 20)).build()

// Definir el evaluador de regresión
val evaluator1 = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Definir el validador cruzado
val cv1 = new CrossValidator()
  .setEstimator(pipeline1)
  .setEvaluator(evaluator1)
  .setEstimatorParamMaps(paramGrid1)
  .setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajustar el modelo utilizando el conjunto de entrenamiento
val cvModel1 = cv1.fit(trainingData)

// Evaluar el modelo en el conjunto de prueba
val predictions1 = cvModel1.transform(testData)
val rmse1 = evaluator1.evaluate(predictions1)



// Imprimir los parámetros del mejor modelo
val bestModel1 = cvModel1.bestModel.asInstanceOf[PipelineModel]
val gbtModel = bestModel1.stages(1).asInstanceOf[GBTRegressionModel]
println(s"Best model parameters: maxDepth = ${gbtModel.getMaxDepth}, maxIter = ${gbtModel.getMaxIter}")



val metricas1 = evaluator1.getMetrics(predictions1)
//Algunas métricas
println(s"MSE: ${metricas1.meanSquaredError}")
println(s"r2: ${metricas1.r2}")
println(s"root MSE: ${metricas1.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas1.meanAbsoluteError}")





















//Validacion cruzada con evaluacion de parametros sobre RandomForest
val rf = new RandomForestRegressor()
  .setLabelCol("cnt")
  .setFeaturesCol("features")

// Crear el pipeline
val pipeline3 = new Pipeline()
  .setStages(Array(assembler, rf))



// Definir el grid de parámetros
val paramGrid3 = new ParamGridBuilder().addGrid(rf.maxDepth, Array(5, 10)).addGrid(rf.numTrees, Array(10, 20)).build()

// Definir el evaluador de regresión
val evaluator3 = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Definir el validador cruzado
val cv3 = new CrossValidator()
  .setEstimator(pipeline3)
  .setEvaluator(evaluator3)
  .setEstimatorParamMaps(paramGrid3)
  .setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajustar el modelo utilizando el conjunto de entrenamiento
val cvModel3 = cv3.fit(trainingData)

// Evaluar el modelo en el conjunto de prueba
val predictions3 = cvModel3.transform(testData)
val rmse3 = evaluator3.evaluate(predictions3)



// Imprimir los parámetros del mejor modelo
val bestModel3 = cvModel3.bestModel.asInstanceOf[PipelineModel]
val rfModel = bestModel3.stages(1).asInstanceOf[RandomForestRegressionModel]
println(s"Best model parameters: maxDepth = ${rfModel.getMaxDepth}, numTrees = ${rfModel.getNumTrees}")

val metricas3 = evaluator3.getMetrics(predictions3)
//Algunas métricas
println(s"MSE: ${metricas3.meanSquaredError}")
println(s"r2: ${metricas3.r2}")
println(s"root MSE: ${metricas3.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas3.meanAbsoluteError}")






















//Validacion cruzada con evaluacion de parametros sobre DecisionTreeRegressor
val dt = new DecisionTreeRegressor()
  .setLabelCol("cnt")
  .setFeaturesCol("features")

// Crear el pipeline
val pipeline2 = new Pipeline()
  .setStages(Array(assembler, dt))



// Definir el grid de parámetros
val paramGrid2 = new ParamGridBuilder().addGrid(dt.maxDepth, Array(5, 10)).addGrid(dt.maxBins, Array(32, 64)).build()

// Definir el evaluador de regresión
val evaluator2 = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Definir el validador cruzado
val cv2 = new CrossValidator()
  .setEstimator(pipeline2)
  .setEvaluator(evaluator2)
  .setEstimatorParamMaps(paramGrid2)
  .setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajustar el modelo utilizando el conjunto de entrenamiento
val cvModel2 = cv2.fit(trainingData)

// Evaluar el modelo en el conjunto de prueba
val predictions2 = cvModel2.transform(testData)
val rmse2 = evaluator2.evaluate(predictions2)



// Imprimir los parámetros del mejor modelo
val bestModel2 = cvModel2.bestModel.asInstanceOf[PipelineModel]
val dtModel = bestModel2.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println(s"Best model parameters: maxDepth = ${dtModel.getMaxDepth}, maxBins = ${dtModel.getMaxBins}")



val metricas2 = evaluator2.getMetrics(predictions2)
//Algunas métricas
println(s"MSE: ${metricas2.meanSquaredError}")
println(s"r2: ${metricas2.r2}")
println(s"root MSE: ${metricas2.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas2.meanAbsoluteError}")


















println(s"RMSe en el conjunto de test del modelo con los mejores parametros para LinearRegression: $rmse")
println(s"RMSe en el conjunto de test del modelo con los mejores parametros para GBTRegressor: $rmse1")
println(s"RMSe en el conjunto de test del modelo con los mejores parametros para DecisionTreeRegressor: $rmse2")
println(s"RMSe en el conjunto de test del modelo con los mejores parametros para RandomForestRegressor: $rmse3")



