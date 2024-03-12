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


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.RandomForestRegressor

// ----------------------------------------------------------------------------------------
// Carga de los datos
// ----------------------------------------------------------------------------------------

println("\nCARGA DE LOS DATOS")

val PATH="/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO="hour.csv"


val bikeDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load(PATH+ARCHIVO)
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

// ----------------------------------------------------------------------------------------
// Transformación de los datos
// ----------------------------------------------------------------------------------------

println("\nTRASNFORMACIÓN DE LOS DATOS")

// Transformación de los atributos categoricos con OneHotEncoder

val indexer = Array("season","yr","mnth","hr","weekday","weathersit").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
val pipeline = new Pipeline().setStages(indexer)
val bikeDF_trans = pipeline.fit(bikeDF).transform(bikeDF).drop("season","yr","mnth","hr","weekday","weathersit")


// ----------------------------------------------------------------------------------------
// Partición de los datos en train y test
// ----------------------------------------------------------------------------------------

println("\nPARTICIÓN DE LOS DATOS")

val splitSeed = 123
val Array(train,test) = bikeDF_trans.randomSplit(Array(0.7,0.3),splitSeed)

//Selección y ensamblado de columna feature

val feature = Array("holiday","workingday","temp","atemp","hum","windspeed","season_Vec","yr_Vec","mnth_Vec","hr_Vec","weekday_Vec","weathersit_Vec")
val assembler = new VectorAssembler().setInputCols(feature).setOutputCol("features")


// ----------------------------------------------------------------------------------------
// Selección de mejor modelo
// ----------------------------------------------------------------------------------------

println("\nSELECCIÓN DE MODELO")


//Modelo de Linear regression
println("\nMODELO LINEAR REGRESSION")

//Construcción del modelo
val lr = new LinearRegression().setLabelCol("cnt").setFeaturesCol("features")
 
//Creamos el pipeline
val pipeline = new Pipeline().setStages(Array(assembler,lr))
 
//Entrenamos modelo
val lrModel = pipeline.fit(train)
val predictions = lrModel.transform(test)
 
//Resultado del modelo
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val metricas = evaluator.getMetrics(predictions)

//Algunas métricas
println(s"MSE: ${metricas.meanSquaredError}")
println(s"r2: ${metricas.r2}")
println(s"root MSE: ${metricas.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas.meanAbsoluteError}")



//Modelo GBT Regressor
println("\nMODELO GBT REGRESSOR")

//Construcción del modelo
val gbt = new GBTRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creamos el pipeline
val pipeline = new Pipeline().setStages(Array(assembler,gbt))
 
//Entrenamos modelo
val gbtModel = pipeline.fit(train)
val predictions = gbtModel.transform(test)
 
//Resultado del modelo
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val metricas = evaluator.getMetrics(predictions)

//Algunas métricas
println(s"MSE: ${metricas.meanSquaredError}")
println(s"r2: ${metricas.r2}")
println(s"root MSE: ${metricas.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas.meanAbsoluteError}")


//Modelo Decision Tree Regressor
println("\nMODELO DECISION TREE REGRESSOR")

//Construcción del modelo
val dt = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creamos el pipeline
val pipeline = new Pipeline().setStages(Array(assembler,dt))
 
//Entrenamos modelo
val dtModel = pipeline.fit(train)
val predictions = dtModel.transform(test)
 
//Resultado del modelo
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val metricas = evaluator.getMetrics(predictions)

//Algunas métricas
println(s"MSE: ${metricas.meanSquaredError}")
println(s"r2: ${metricas.r2}")
println(s"root MSE: ${metricas.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas.meanAbsoluteError}")


//Modelo Random Forest Regressor
println("\nMODELO RANDOM FOREST REGRESSOR")

//Construcción del modelo
val rf = new RandomForestRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creamos el pipeline
val pipeline = new Pipeline().setStages(Array(assembler,rf))
 
//Entrenamos modelo
val rfModel = pipeline.fit(train)
val predictions = rfModel.transform(test)
 
//Resultado del modelo
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val metricas = evaluator.getMetrics(predictions)

//Algunas métricas
println(s"MSE: ${metricas.meanSquaredError}")
println(s"r2: ${metricas.r2}")
println(s"root MSE: ${metricas.rootMeanSquaredError}")
println(s"Mean Absolute Error: ${metricas.meanAbsoluteError}")

// ----------------------------------------------------------------------------------------
// Guardamos el mejor modelo (GBT Regressor)
// ----------------------------------------------------------------------------------------

println("\nGUARDAMOS EL MEJOR MODELO (GBT Regressor)")


//Guardamos el mejor modelo
gbtModel.write.overwrite().save(PATH+"/modelo")

