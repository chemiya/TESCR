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

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.util.IntParam
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.log4j._
import org.apache.spark.sql.functions.to_timestamp
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.RandomForestRegressor



val PATH="/home/usuario/Regresion/MiniProyecto2/"
val ARCHIVO="hour.csv"

// ----------------------------------------------------------------------------------------
// Carga de los datos
// ----------------------------------------------------------------------------------------

println("\nCARGA DE LOS DATOS")


val bikeDF = spark.read.format("csv").option("inferSchema",true).option("header",true).load(PATH+ARCHIVO)
bikeDF.show(10)

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





// Transformación de los atributos categoricos

val indexer = Array("season","weathersit").map(c=>new OneHotEncoder().setInputCol(c).setOutputCol(c + "_Vec"))
val pipeline = new Pipeline().setStages(indexer)
val df_r = pipeline.fit(bikeDF).transform(bikeDF).drop("season","weathersit")




val splitSeed = 123
val Array(train,train_test) = df_r.randomSplit(Array(0.7,0.3),splitSeed)

//Generate Feature Column
val feature = Array("holiday","workingday","weekday","temp","atemp","hum","windspeed","season_Vec","weathersit_Vec","yr","mnth","hr")
//Assemble Feature Column
val assembler = new VectorAssembler().setInputCols(feature).setOutputCol("features")


//Linear Regression Model


//Model Building
val lr = new LinearRegression().setLabelCol("cnt").setFeaturesCol("features")
 
//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,lr))
 
//Training Model
val lrModel = pipeline.fit(train)
val predictions = lrModel.transform(train_test)
 
//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Linear Regression Root Mean Squared Error (RMSE) on train_test data = " + rmse)



//GBT Regressor

//Model Building
val gbt = new GBTRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creating pipeline
val pipeline = new Pipeline().setStages(Array(assembler,gbt))
 
//Training Model
val gbtModel = pipeline.fit(train)
val predictions = gbtModel.transform(train_test)
 
//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("GBT Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)


//Decision Tree Regressor

//Model Building
val dt = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,dt))
 
//Training Model
val dtModel = pipeline.fit(train)
val predictions = dtModel.transform(train_test)
 
//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Decision Tree Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)


//Random Forest Regressor

//Model Building
val rf = new RandomForestRegressor().setLabelCol("cnt").setFeaturesCol("features")
 
//Creating Pipeline
val pipeline = new Pipeline().setStages(Array(assembler,rf))
 
//Training Model
val rfModel = pipeline.fit(train)
val predictions = rfModel.transform(train_test)
 
//Model Summary
val evaluator = new RegressionEvaluator().setLabelCol("cnt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Random Forest Regressor Root Mean Squared Error (RMSE) on train_test data = " + rmse)


