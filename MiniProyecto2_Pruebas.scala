/*
    Master en Ingeniería Informática - Universidad de Valladolid

    TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
    Mini-Proyecto 1 : Recomendadores

    Dataset: CDs and Vinyl

    Segundo script

    Grupo 2:
    Sergio Agudelo Bernal
    Miguel Ángel Collado Alonso
    José María Lozano Olmedo
*/

// Iniciar sesión como $> spark-shell --driver-memory 4g
// Indispensable haber ejecutado el primer script!


// -----------------------------------------------------------------------------
// Configuraciones iniciales
// -----------------------------------------------------------------------------

// Mostrar solo errores en consola
sc.setLogLevel("ERROR")

// Importación de módulos a usar
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionSummary}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}


// -----------------------------------------------------------------------------
// Carga de los datos
// -----------------------------------------------------------------------------

// Variables de ruta del archivo de datos
val PATH = "/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO_TEST = "testData"

val testRaw = spark.read.format("csv").
    option("inferSchema", true).
    load(PATH + ARCHIVO_TEST).
    toDF(
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "cnt"
    )

val featureCols = Array(
    "holiday",
    "workingday",
    "temp",
    "hum",
    "season",
    "yr",
    "mnth",
    "hr",
    "weekday",
    "weathersit"
)

val assembler = new VectorAssembler().
    setInputCols(featureCols).
    setOutputCol("features")

val testData = assembler.transform(testRaw)

testData.show(5)

// -----------------------------------------------------------------------------
// Evaluación de modelo LinearRegressionModel
// -----------------------------------------------------------------------------

println("\nEVALUACIÓN DE MODELO LinearRegressionModel")

// Carga del modelo
val lrModel = LinearRegressionModel.load(PATH + "best_LinearRegressionModel")

// Definición del evaluador de regresión
val evaluator = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Evaluación del modelo bajo conjunto de prueba
val predictions = lrModel.transform(testData)
val rmse = evaluator.evaluate(predictions)

// Visualización de métricas para el mejor modelo
val metrics = evaluator.getMetrics(predictions)
println(s"MSE: ${metrics.meanSquaredError}")
println(s"R²: ${metrics.r2}")
println(s"root MSE: ${metrics.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics.meanAbsoluteError}")


// -----------------------------------------------------------------------------
// Evaluación de modelo GBTRegressionModel
// -----------------------------------------------------------------------------

println("\nEVALUACIÓN DE MODELO GBTRegressionModel")

// Carga del modelo
val gbtModel = GBTRegressionModel.load(PATH + "best_GBTRegressionModel")

// Definición del evaluador de regresión
val evaluator1 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Evaluación del modelo bajo conjunto de prueba
val predictions1 = gbtModel.transform(testData)
val rmse1 = evaluator1.evaluate(predictions1)

// Visualización de métricas para el mejor modelo
val metrics1 = evaluator1.getMetrics(predictions1)
println(s"MSE: ${metrics1.meanSquaredError}")
println(s"R²: ${metrics1.r2}")
println(s"root MSE: ${metrics1.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics1.meanAbsoluteError}")


// -----------------------------------------------------------------------------
// Evaluación de modelo RandomForestRegressionModel
// -----------------------------------------------------------------------------

println("\nEVALUACIÓN DE MODELO RandomForestRegressionModel")

// Carga del modelo
val rfModel = RandomForestRegressionModel.
    load(PATH + "best_RandomForestRegressionModel")

// Definición del evaluador de regresión
val evaluator3 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Evaluación del modelo en el conjunto de prueba
val predictions3 = rfModel.transform(testData)
val rmse3 = evaluator3.evaluate(predictions3)

// Visualización de métricas para el mejor modelo
val metrics3 = evaluator3.getMetrics(predictions3)
println(s"MSE: ${metrics3.meanSquaredError}")
println(s"R²: ${metrics3.r2}")
println(s"root MSE: ${metrics3.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics3.meanAbsoluteError}")


// -----------------------------------------------------------------------------
// Evaluación de modelo DecisionTreeRegressor
// -----------------------------------------------------------------------------

println("\nEVALUACIÓN DE MODELO DecisionTreeRegressor")

// Carga del modelo
val dtModel = DecisionTreeRegressionModel.
    load(PATH + "best_DecisionTreeRegressionModel")

// Definición del evaluador de regresión
val evaluator2 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Evaluación del modelo en el conjunto de prueba
val predictions2 = dtModel.transform(testData)
val rmse2 = evaluator2.evaluate(predictions2)

// Visualización de métricas para el mejor modelo
val metrics2 = evaluator2.getMetrics(predictions2)
println(s"MSE: ${metrics2.meanSquaredError}")
println(s"R²: ${metrics2.r2}")
println(s"root MSE: ${metrics2.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metrics2.meanAbsoluteError}")


// -----------------------------------------------------------------------------
// Selección del mejor modelo
// -----------------------------------------------------------------------------

println("\nSELECCIÓN DEL MEJOR MODELO")

println(s"RMSE en el conjunto de test para mejor modelo de LinearRegression: ${metrics.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de GBTRegressor: ${metrics1.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de DecisionTreeRegressor: ${metrics2.rootMeanSquaredError}")
println(s"RMSE en el conjunto de test para mejor modelo de RandomForestRegressor: ${metrics3.rootMeanSquaredError}")

println("\nGUARDADO DEL MEJOR MODELO: GBTRegressor")

gbtModel.write.overwrite().save(PATH + "modelo")

// -----------------------------------------------------------------------------
// Evaluación del mejor modelo
// -----------------------------------------------------------------------------

println("\nEVALUACIÓN DEL MEJOR MODELO (GBTRegressionModel)")

// Carga del modelo
val bestModel = GBTRegressionModel.load(PATH + "modelo")

// Definición del evaluador de regresión
val bestEvaluator = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Evaluación del modelo bajo conjunto de prueba
val bestPredictions = bestModel.transform(testData)
val bestRmse = bestEvaluator.evaluate(bestPredictions)

// Visualización de métricas para el mejor modelo
val bestMetrics = bestEvaluator.getMetrics(bestPredictions)
println(s"MSE: ${bestMetrics.meanSquaredError}")
println(s"R²: ${bestMetrics.r2}")
println(s"root MSE: ${bestMetrics.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${bestMetrics.meanAbsoluteError}")