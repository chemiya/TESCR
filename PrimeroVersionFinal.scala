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

// Arrancar spark-shell como: $ spark-shell --driver-memory 8g --executor-memory 8g --executor-cores 4

// Configuraciones iniciales
sc.setCheckpointDir("checkpoint")
sc.setLogLevel("ERROR")

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
case class Rating( item: String, user: String, rating: Float, timestamp: Int )

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

var dfCDsVinylTransformado = dfCDsVinyl.
    withColumn("itemId", col("itemId").cast("Int")).
    withColumn("userId", col("userId").cast("Int")).
    select("itemId", "item", "userId", "user", "rating", "timestamp")

println("Conjunto de datos transformado:")
dfCDsVinylTransformado.show(5)

// Estos son los "diccionarios" que se podrán usar para buscar user e item por sus id respectivos
dfCDsVinylTransformado.
    select("user", "userId").
    distinct().
    write.mode("overwrite").
    csv(PATH + "dfUserLookup")

dfCDsVinylTransformado.
    select("item", "itemId").
    distinct().
    write.mode("overwrite").
    csv(PATH + "dfItemLookup")

// ----------------------------------------------------------------------------------------
// Creación de conjuntos training y test
// ----------------------------------------------------------------------------------------

println("\nCREACIÓN DE CONJUNTOS TRAINING Y TEST")

// Factor de escala para estratificar el conjunto de entrenamiento

// SCALE_FACTOR = .0025 resulta en un volumen de ~10k registros
// Validación cruzada tarda cerca de

val SCALE_FACTOR = .8

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

// Generamos el dataset de test eliminando los elementos de training
val test = dfCDsVinylTransformado.except(training)
println("Registros test: " + test.count())

test.write.mode("overwrite").csv(PATH + "conjunto-test")
println("Conjunto de test guardado")

println("Registros training antes de limpieza: " + training.count())

// Limpieza - Estrategias 1 y 2
// Próximos pasos en el miniproyecto: Encontrar una forma óptima de hacer el algoritmo,
// con implementación en Spark o SQL, por ejemplo.

// println("Comienza comprobación de los datos de los usuarios")
// val valoresUnicosUsuario = training.select("userId").distinct().collect()
// val usuariosDiferentes=valoresUnicosUsuario.length
// var contador=0

// // Recorrido diferentes usuarios
// for (fila <- valoresUnicosUsuario) {
//     println(s"Evaluando datos de: $contador / $usuariosDiferentes")
//     contador=contador+1
//     val valorUnicoDouble = fila.getDouble(0)

//     // Filtro opiniones de cada usuario
//     val dfFiltrado = training.filter(col("userId") === valorUnicoDouble)

//     // Cuántas opiniones han hecho
//     val filas=dfFiltrado.count()

//     // Desviación estándar de los timestamp de sus opiniones
//     val desviacionEstandar = dfFiltrado.
//         agg(stddev("timestamp").
//         alias("desviacion_estandar")).
//         collect()(0).
//         getAs[Double]("desviacion_estandar")

//     // Condicional para eliminar datos si cumple condiciones
//     if(desviacionEstandar < 31622400 && filas > 50) {
//         training = training.filter(col("userId") =!= valorUnicoDouble)
//         val numeroFilas = training.count()
//         println(s"Se han eliminado los registros del usuario por no cumplir condición 1. El número de filas en el DataFrame es: $numeroFilas")
//     }

//     // Cuántas opiniones con cada rating han hecho
//     var cuentaOpiniones = dfFiltrado.
//         groupBy("rating").
//         count().
//         orderBy(desc("count")).
//         withColumnRenamed("count", "cuenta")

//     // Convertimos en porcentaje
//     cuentaOpiniones = cuentaOpiniones.
//         select( format_number((col("cuenta") / filas) * 100, 2).alias("Resultado") )

//     // Porcentaje opiniones rating 5
//     val primerValorPorcentaje = cuentaOpiniones.
//         select("Resultado").
//         first().
//         getAs[String](0)

//     val valorDouble = primerValorPorcentaje.toDouble

//     // Condicional para eliminar datos si cumple condiciones
//     if(valorDouble > 90 && filas > 50) {
//         training = training.filter( col("userId") =!= valorUnicoDouble )
//         val numeroFilas = training.count()
//         println(s"Se han eliminado los registros del usuario por no cumplir condicion 2. El número de filas en el DataFrame es: $numeroFilas")
//     }
// }

// Limpieza - Estrategia 3: Dejar solo la calificación más reciente que el usuario hizo a determinado ítem
training.createOrReplaceTempView("training")

training = spark.sql("""
WITH CTE AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY userId, itemId ORDER BY timestamp DESC) AS rating_order
  FROM training
)
SELECT * FROM CTE WHERE rating_order = 1;
""").drop("rating_order")

println("Registros training después de limpieza: " + training.count())

// Eliminación de columnas no usadas en conjunto de training
training = training.select("itemId", "userId", "rating")

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
    addGrid(als.rank, Array(2)).
    addGrid(als.regParam, Array(0.01, 0.1)).
    addGrid(als.maxIter, Array(5, 7)).
    addGrid(als.alpha, Array(0.01)).
    build()

// Creación de evaluador por métrica RMSE
val evaluator = new RegressionEvaluator().
    setMetricName("rmse").
    setLabelCol("rating").
    setPredictionCol("prediction")

// Creación validador cruzado
val cv1 = new CrossValidator().
    setEstimator(als).
    setEstimatorParamMaps(paramGrid).
    setEvaluator(evaluator).
    setNumFolds(2).
    setParallelism(2)

// Inicio validación cruzada
println(s"Inicio: ${Calendar.getInstance.getTime}")
val cvmodel1 = cv1.fit(training)
println(s"Fin: ${Calendar.getInstance.getTime}")

// ----------------------------------------------------------------------------------------
// Guardado del mejor modelo
// ----------------------------------------------------------------------------------------

println("\nGUARDADO DEL MEJOR MODELO")

// Selección de mejor modelo
val model = cvmodel1.bestModel.asInstanceOf[ALSModel]

println(s"Mejor valor para 'rank': ${model.rank}")
// Las otras métricas no pueden ser visualizadas por el bestModel ser instancia de ALSModel y no de ALS
// https://stackoverflow.com/a/38048635
// En la documentación vemos que no existen getters de hiperparámetros
// https://spark.apache.org/docs/3.5.0/api/scala/org/apache/spark/ml/recommendation/ALSModel.html
// Adicionalmente, la sentencia "cvmodel1.bestModel.asInstanceOf[ALS]" da error.

// println(s"Mejor valor para 'maxIter': ${model.getMaxIter()}")
// println(s"Mejor valor para 'regParam': ${model.getRegParam()}")
// println(s"Mejor valor para 'alpha': ${model.getAlpha()}")

model.write.overwrite().save(PATH + "modeloALS")
println("Modelo guardado")

// ----------------------------------------------------------------------------------------
// Validación del modelo
// ----------------------------------------------------------------------------------------

println("\nVALIDACIÓN DEL MODELO")

// Configuración de estrategia y generación de predicciones
model.setColdStartStrategy("drop")
val predictions = model.transform(training)

// Cálculo del error cuadrático medio
val rmse = evaluator.evaluate(predictions)
println(s"Error cuadrático medio: $rmse")