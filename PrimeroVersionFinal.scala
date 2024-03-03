/*
    Master en Ingeniería Informática - Universidad de Valladolid

    TECNICAS ESCLABLES DE ANÁLISIS DE DATOS EN ENTORNOS BIG DATA: Regresión y descubrimiento de conocimiento
    Mini-Proyecto 1 : Recomendadores

    Dataset: CDs and Vinyl

    Primer script

    Grupo 2:
    Sergio Agudelo Bernal
    Miguel Ángel Collado Alonso
    José María Lozano Olmedo
*/

// iniciar sesión como $> spark-shell --driver-memory 4g

// Configuraciones iniciales
// sc.setCheckpointDir("../checkpoint")

// Importación de módulos a usar
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Calendar
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

// Variables de ruta del archivo de datos
val PATH = "/home/usuario/Regresion/"
val ARCHIVO = "CDs_and_Vinyl.csv"


// ----------------------------------------------------------------------------------------
// Carga de los datos
// ----------------------------------------------------------------------------------------

println("\nCARGA DE LOS DATOS")

// Definición del esquema de datos
// Formato del archivo de datos: item,user,rating,timestamp separados por coma

// timestamp es tipo Int porque su rango no supera el valor máximo posible del tipo de dato: 2147483647 (marca de tiempo del 2038-01-19 04:14:07)
case class Rating( item: String, user: String, rating: Float, timestamp: Int)

def parseRating(str: String): Rating = {
    val fields = str.split(",")
    assert(fields.size == 4)

    Rating(
        fields(0).toString,
        fields(1).toString,
        fields(2).toFloat,
        fields(3).toInt
    )
}

val ratings = spark.
    read.textFile(PATH + ARCHIVO).
    map(parseRating).
    toDF()

// Eliminamos nulos
var dfCDsVinyl = ratings.na.drop()


// ----------------------------------------------------------------------------------------
// Exploración
// ----------------------------------------------------------------------------------------

println("\nEXPLORACIÓN")

println("Número total de registros: " + dfCDsVinyl.count())

println("Primeros 5 registros:")
dfCDsVinyl.show(5)

println("Estructura del DataFrame:")
dfCDsVinyl.printSchema()

println("Resumen estadístico de Rating:")
dfCDsVinyl.describe("rating").show()

println("Número de votos por valor:")
dfCDsVinyl.
    groupBy("rating").
    count().
    orderBy(desc("count")).
    withColumnRenamed("count", "cuenta").
    show()

println("Rango de timestamp:")
val MinMaxTime = dfCDsVinyl.agg(min("timestamp"), max("timestamp")).head()

// Convertir el timestamp a formato de fecha
val dateFormat = new SimpleDateFormat("dd-MM-yyyy hh:mm")

// Multiplicamos por 1000L porque Timestamp espera milisegundos
val minFechaStr = dateFormat.format( new Timestamp(MinMaxTime.getInt(0) * 1000L) )
val maxFechaStr = dateFormat.format( new Timestamp(MinMaxTime.getInt(1) * 1000L) )

println("Mínimo valor de timestamp: " + MinMaxTime(0) + " -> " +  minFechaStr)
println("Máximo valor de timestamp: " + MinMaxTime(1) + " -> " +  maxFechaStr)

println("Número de productos (Item): " + dfCDsVinyl.select("item").distinct.count())
println("Número de usuarios (User): " + dfCDsVinyl.select("user").distinct.count())

// ----------------------------------------------------------------------------------------
// Transformaciones
// ----------------------------------------------------------------------------------------

println("\nTRANSFORMACIONES")

// Creación de StringIndexer para asignar a columnas user e item un índice entero
val columnasIndexer = Seq("item", "user")

val indexadores = columnasIndexer.map {
    columna =>

    new StringIndexer()
        .setInputCol(columna)
        .setOutputCol(s"${columna}Id")
}

indexadores.foreach {
    indexador =>

    dfCDsVinyl = indexador.
        fit(dfCDsVinyl).
        transform(dfCDsVinyl)
}

dfCDsVinyl = dfCDsVinyl.
    withColumn("itemId", col("itemId").cast("Int")).
    withColumn("userId", col("userId").cast("Int"))

// Estos son los "diccionarios" que se podrán usar para buscar user e item por sus id respectivos
val dfUserLookup = dfCDsVinyl.select("user", "userId").distinct()
val dfItemLookup = dfCDsVinyl.select("item", "itemId").distinct()


// NOTA: Si esta transformación va, hay que buscar una opción que no sea ciclo for!!!
// (p.ej., filtros y agregaciones en Spark o Spark SQL)
/*
//algoritmo eliminar opiniones falsas y sesgadas-------------------

println("Comienza comprobacion de los datos de los usuarios")
val valoresUnicosUsuario = dfCDsVinylTransformado.select("userId").distinct().collect()
val usuariosDiferentes=valoresUnicosUsuario.length
var contador=0

//recorro diferentes usuarios
for (fila <- valoresUnicosUsuario) {
  println(s"Evaluando datos de: $contador / $usuariosDiferentes")
  contador=contador+1
  val valorUnicoDouble = fila.getDouble(0)

  //filtro opiniones de cada usuario
  val dfFiltrado = dfCDsVinylTransformado.filter(col("userId") === valorUnicoDouble)
  //cuantas opiniones han hecho
  val filas=dfFiltrado.count()
  //desviacion estandar de los timestamp de sus opiniones
  val desviacionEstandar = dfFiltrado.agg(stddev("timestamp").alias("desviacion_estandar")).collect()(0).getAs[Double]("desviacion_estandar")


  //condicional para eliminar datos si cumple condiciones
  if(desviacionEstandar<31622400 && filas>50){
    dfCDsVinylTransformado= dfCDsVinylTransformado.filter(col("userId") =!= valorUnicoDouble)
    val numeroFilas = dfCDsVinylTransformado.count()
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
    dfCDsVinylTransformado= dfCDsVinylTransformado.filter(col("userId") =!= valorUnicoDouble)
    val numeroFilas = dfCDsVinylTransformado.count()
    println(s"Se han eliminado los registros del usuario por no cumplir condicion 2. El número de filas en el DataFrame es: $numeroFilas")
  }

}
*/

var dfCDsVinylTransformado = dfCDsVinyl.
    select("itemId", "userId", "rating").
    repartition(100)

println("Conjunto de datos transformado:")
dfCDsVinylTransformado.show(5)

// ----------------------------------------------------------------------------------------
// Creación de conjuntos training y test
// ----------------------------------------------------------------------------------------

println("\nCREACIÓN DE CONJUNTOS TRAINING Y TEST")

// Antes:
// val Array(training, test) = dfCDsVinylTransformado.randomSplit(Array(0.8, 0.2))

// Factor de escala para estratificar el conjunto de entrenamiento

// SCALE_FACTOR = .0025 resulta en un volumen de ~10k registros
// Validación cruzada tarda cerca de

val SCALE_FACTOR = .0025

var training = dfCDsVinylTransformado.
    withColumn("ratingId", col("rating").cast("String")).
    stat.sampleBy("ratingId",
        fractions = Map(
            "1.0" -> SCALE_FACTOR,
            "2.0" -> SCALE_FACTOR,
            "3.0" -> SCALE_FACTOR,
            "4.0" -> SCALE_FACTOR,
            "5.0" -> SCALE_FACTOR
), seed = 10).drop("ratingId")

println("Cuenta de filas del conjunto de entrenamiento: " + training.count())

val test = dfCDsVinylTransformado.except(training)

test.write.mode("overwrite").csv(PATH + "conjunto-test")
println("Conjunto de test guardado")

// ----------------------------------------------------------------------------------------
// Validación cruzada
// ----------------------------------------------------------------------------------------

println("\nVALIDACIÓN CRUZADA")

// Creación modelo ALS
val als = new ALS().
    setUserCol("userId").
    setItemCol("itemId").
    setRatingCol("rating")

// Creación de "grid" de hiperparámetros
//val paramGrid = new ParamGridBuilder().addGrid(als.rank, Array(2,3,5)).addGrid(als.regParam, Array(0.01, 0.1,0.2)).addGrid(als.maxIter, Array(5,7,10)).build()
val paramGrid = new ParamGridBuilder().
    addGrid(als.rank, Array(2, 3)).
    addGrid(als.regParam, Array(0.01, 0.1)).
    addGrid(als.maxIter, Array(5, 7)).
    addGrid(als.alpha, Array(0.01)).
    build()

// Creación de evaluador por métrica RMSE
val evaluator = new RegressionEvaluator()
evaluator.setMetricName("rmse")
evaluator.setLabelCol("rating")
evaluator.setPredictionCol("prediction")

// Creación validador cruzado
val cv1 = new CrossValidator().
    setParallelism(4).
    setEstimator(als).
    setEstimatorParamMaps(paramGrid).
    setEvaluator(evaluator).
    setNumFolds(1)

// Inicio validación cruzada
println(s"Inicio: ${Calendar.getInstance.getTime}")

spark.time {
    val cvmodel1 = cv1.fit(training)
}

println(s"Fin: ${Calendar.getInstance.getTime}")

val model = als.fit(training)

// ----------------------------------------------------------------------------------------
// Guardado del mejor modelo
// ----------------------------------------------------------------------------------------

println("\nGUARDADO DEL MEJOR MODELO")

// Selección de mejor modelo
val bestModel = cvmodel1.bestModel
val model = bestModel.asInstanceOf[ALSModel]

println(s"Mejor valor para 'rank': ${model.rank}")
println(s"Mejor valor para 'maxIter': ${model.maxIter}")
println(s"Mejor valor para 'regParam': ${model._java_obj.getRegParam()}")
println(s"Mejor valor para 'alpha': ${model._java_obj.getAlpha()}")

model.write.overwrite().save(PATH + "modeloALS")

// ----------------------------------------------------------------------------------------
// Validación del modelo
// ----------------------------------------------------------------------------------------

println("\nVALIDACIÓN DEL MODELO")