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

// Iniciar sesión como $> spark-shell --driver-memory 4g


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

println("\nCARGA DE LOS DATOS")

val PATH = "/home/usuario/Regresion/MiniProyecto2_Sergio/"
val ARCHIVO = "hour.csv"

val bikeDF = spark.read.format("csv").
    option("inferSchema", true).
    option("header", true).
    load(PATH + ARCHIVO)

println("Datos cargados:")
bikeDF.show(10)


// -----------------------------------------------------------------------------
// Exploración de los datos
// -----------------------------------------------------------------------------

println("\nEXPLORACIÓN DE LOS DATOS")

// Comprobación del esquema de datos
bikeDF.printSchema

// Comprobación de valores nulos
bikeDF.
    select(
        bikeDF.columns.map(
            c => sum(col(c).isNull.cast("int")).alias(c)): _*
    ).show()

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

// Eliminar atributos y borrar columnas vacías
val columnasAEliminar = Seq(
    "instant",
    "dteday",
    "atemp",
    "windspeed",
    "casual",
    "registered"
)

val nuevoDF = bikeDF.drop(columnasAEliminar: _*)
nuevoDF.count()


// -----------------------------------------------------------------------------
// Transformación de los datos - Ya no se usa
// -----------------------------------------------------------------------------

// println("\nTRASNFORMACIÓN DE LOS DATOS")

// // Transformación de los atributos categoricos con OneHotEncoder

// val indexer = Array("season","yr","mnth","hr","weekday","weathersit").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
// val pipeline = new Pipeline().setStages(indexer)
// val bikeDF_trans = pipeline.fit(nuevoDF).transform(nuevoDF).drop("season","yr","mnth","hr","weekday","weathersit")
// bikeDF_trans.show()


// -----------------------------------------------------------------------------
// Partición de los datos para entrenamiento y pruebas
// -----------------------------------------------------------------------------

println("\nPARTICIÓN DE LOS DATOS")

val splitSeed = 123
val Array(trainingData, testData) = nuevoDF.
    randomSplit(Array(0.7, 0.3), splitSeed)

testData.write.mode("overwrite").csv(PATH + "testData")
println("Conjunto de pruebas guardado")

testData.show(5)

//Selección y ensamblado de columna feature
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


// -----------------------------------------------------------------------------
// Validación cruzada para Regresión Lineal
// -----------------------------------------------------------------------------

println("\nVALIDACIÓN CRUZADA PARA REGRESIÓN LINEAL")

// Definición del modelo
val lr = new LinearRegression().
    setLabelCol("cnt").
    setFeaturesCol("features")

// Creación de pipeline
val pipeline = new Pipeline().
    setStages(Array(assembler, lr))

// Definición de grid de parámetros
val paramGrid = new ParamGridBuilder().
    // Anterior
    // addGrid(lr.regParam, Array(0.1, 0.01)).
    // addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).
    // Primera iteración
    // addGrid(lr.regParam, Array(0.01, 1.01, 2.01, 3.01, 4.01, 5.01)).
    // addGrid(lr.elasticNetParam, Array(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)).
    // Segunda iteración
    // addGrid(lr.regParam, Array(0.01, 0.21, 0.41, 0.61, 0.81, 1.01)).
    // addGrid(lr.elasticNetParam, Array(0.8, 0.84, 0.8, 0.92, 0.96, 1.0)).
    // Tercera iteración
    addGrid(lr.regParam, Array(0.13, 0.17, 0.21, 0.25, 0.29, 0.33)).
    addGrid(lr.elasticNetParam, Array(0.784, 0.792, 0.8, 0.808, 0.888, 0.968)).
    build()

// Definición del evaluador de regresión
val evaluator = new RegressionEvaluator()
  .setLabelCol("cnt")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

// Definición del validador cruzado
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajuste del modelo utilizando el conjunto de entrenamiento
val cvModel = cv.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESIÓN LINEAL")

// Visualización de los parámetros del mejor modelo
val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
val lrModel = bestModel.stages(1).asInstanceOf[LinearRegressionModel]
println(s"""Parámetros del mejor modelo:
regParam = ${lrModel.getRegParam}, elasticNetParam = ${lrModel.getElasticNetParam}
""")

// Parámetros del mejor modelo:
// regParam = 0.13, elasticNetParam = 0.808
lrModel.write.overwrite().save(PATH + "best_LinearRegressionModel")
println("Mejor modelo regresión lineal guardado")


// -----------------------------------------------------------------------------
// Validación cruzada con regresor GBT
// -----------------------------------------------------------------------------

println("\nVALIDACIÓN CRUZADA PARA REGRESOR GBT")

// Definición del modelo
val gbt = new GBTRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")

// Creación de pipeline
val pipeline1 = new Pipeline().
    setStages(Array(assembler, gbt))

// Definición de grid de parámetros
val paramGrid1 = new ParamGridBuilder().
    // Anterior
    // addGrid(gbt.maxDepth, Array(5, 10)).
    // addGrid(gbt.maxIter, Array(10, 20)).
    // Primera iteración
    // addGrid(gbt.maxDepth, Array(2, 6, 10)).
    // addGrid(gbt.maxIter, Array(1, 10, 20)).
    // Segunda iteración
    // addGrid(gbt.maxDepth, Array(4, 6, 8)).
    // addGrid(gbt.maxIter, Array(10, 15, 20)).
    // Tercera iteración
    addGrid(gbt.maxDepth, Array(5, 6, 7)).
    addGrid(gbt.maxIter, Array(16, 18, 20)).
    build()

// Definición del evaluador de regresión
val evaluator1 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Definición del validador cruzado
val cv1 = new CrossValidator().
    setEstimator(pipeline1).
    setEvaluator(evaluator1).
    setEstimatorParamMaps(paramGrid1).
    setNumFolds(3) // Número de pliegues para la validación cruzada

// Ajuste del modelo utilizando el conjunto de entrenamiento
val cvModel1 = cv1.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR GBT")

// Visualización de los parámetros del mejor modelo
val bestModel1 = cvModel1.bestModel.asInstanceOf[PipelineModel]
val gbtModel = bestModel1.stages(1).asInstanceOf[GBTRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${gbtModel.getMaxDepth}, maxIter = ${gbtModel.getMaxIter}
""")

// Parámetros del mejor modelo:
// maxDepth = 7, maxIter = 20
gbtModel.write.overwrite().save(PATH + "best_GBTRegressionModel")
println("Mejor modelo regresor GBT guardado")


// -----------------------------------------------------------------------------
// Validación cruzada con regresor RandomForest
// -----------------------------------------------------------------------------

println("\nVALIDACIÓN CRUZADA PARA REGRESOR RF")

//Validación cruzada con evaluación de parámetros sobre RandomForest
val rf = new RandomForestRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")

// Creación de pipeline
val pipeline3 = new Pipeline().
    setStages(Array(assembler, rf))

// Definición del grid de parámetros
val paramGrid3 = new ParamGridBuilder().
    // Anterior
    // addGrid(rf.maxDepth, Array(5, 10)).
    // addGrid(rf.numTrees, Array(10, 20)).
    // Primera iteración
    // addGrid(rf.maxDepth, Array(1, 5, 10)).
    // addGrid(rf.numTrees, Array(10, 20, 30)).
    // Segunda iteración
    // addGrid(rf.maxDepth, Array(8, 10, 12)).
    // addGrid(rf.numTrees, Array(5, 10, 15)).
    // Tercera iteración
    addGrid(rf.maxDepth, Array(11, 12, 13)).
    addGrid(rf.numTrees, Array(13, 15, 17)).
    build()

// Definición del evaluador de regresión
val evaluator3 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Definición del validador cruzado
val cv3 = new CrossValidator().
    setEstimator(pipeline3).
    setEvaluator(evaluator3).
    setEstimatorParamMaps(paramGrid3).
    setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajuste del modelo utilizando el conjunto de entrenamiento
val cvModel3 = cv3.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR RF")

// Visualización de parámetros del mejor modelo
val bestModel3 = cvModel3.bestModel.asInstanceOf[PipelineModel]
val rfModel = bestModel3.stages(1).asInstanceOf[RandomForestRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${rfModel.getMaxDepth}, numTrees = ${rfModel.getNumTrees}
""")

// Parámetros del mejor modelo:
// maxDepth = 13, numTrees = 15
rfModel.write.overwrite().save(PATH + "best_RandomForestRegressionModel")
println("Mejor modelo regresor RF guardado")


// -----------------------------------------------------------------------------
// Validación cruzada con regresor DT
// -----------------------------------------------------------------------------

println("\nVALIDACIÓN CRUZADA PARA REGRESOR DT")

// Validación cruzada con evaluación de parámetros sobre DecisionTreeRegressor
val dt = new DecisionTreeRegressor().
    setLabelCol("cnt").
    setFeaturesCol("features")

// Creación de pipeline
val pipeline2 = new Pipeline().
    setStages(Array(assembler, dt))

// Definición de grid de parámetros
val paramGrid2 = new ParamGridBuilder().
    // Anterior
    // addGrid(dt.maxDepth, Array(5, 10)).
    // addGrid(dt.maxBins, Array(32, 64)).
    // Primera iteración
    // addGrid(dt.maxDepth, Array(1, 5, 10)).
    // addGrid(dt.maxBins, Array(16, 32, 64)).
    // Segunda iteración
    // addGrid(dt.maxDepth, Array(8, 10, 12)).
    // addGrid(dt.maxBins, Array(32, 64, 96)).
    // Tercera iteración
    addGrid(dt.maxDepth, Array(11, 12, 13)).
    addGrid(dt.maxBins, Array(16, 32, 48)).
    build()

// Definición del evaluador de regresión
val evaluator2 = new RegressionEvaluator().
    setLabelCol("cnt").
    setPredictionCol("prediction").
    setMetricName("rmse")

// Definición del validador cruzado
val cv2 = new CrossValidator().
    setEstimator(pipeline2).
    setEvaluator(evaluator2).
    setEstimatorParamMaps(paramGrid2).
    setNumFolds(5) // Número de pliegues para la validación cruzada

// Ajuste del modelo utilizando el conjunto de entrenamiento
val cvModel2 = cv2.fit(trainingData)

println("\nPARÁMETROS MEJOR MODELO REGRESOR DT")

// Visualización de los parámetros del mejor modelo
val bestModel2 = cvModel2.bestModel.asInstanceOf[PipelineModel]
val dtModel = bestModel2.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println(s"""Parámetros del mejor modelo:
maxDepth = ${dtModel.getMaxDepth}, maxBins = ${dtModel.getMaxBins}
""")

// Parámetros del mejor modelo:
// maxDepth = 12, maxBins = 32
dtModel.write.overwrite().save(PATH + "best_DecisionTreeRegressionModel")
println("Mejor modelo regresor DT guardado")