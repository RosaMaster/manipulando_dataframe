from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, arrays_zip, struct, array
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType

# 1. Inicializar a SparkSession
# Isso é o ponto de entrada para qualquer funcionalidade do PySpark
spark = SparkSession.builder \
    .appName("FlattenNestedJSON") \
    .getOrCreate()

# 2. Seus dados originais
# No PySpark, é comum representar os dados brutos como uma lista de dicionários Python
# antes de criar o DataFrame Spark.
data_raw = [
    {
        'empresa': 'Viagem e Turismo Ltda',
        'cnpj': '12345678000190',
        'meioContato': {
            'contatos': [
                {
                    'id': 1,
                    'tipo': 'email',
                    'valor': 'email1@gmail.com',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'geral', 'status': 'ativo'},
                        {'nome': 'atendimento', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 2,
                    'tipo': 'telefone',
                    'valor': '11988886666',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'comercial', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 3,
                    'tipo': 'email',
                    'valor': 'email2@gmail.com',
                    'status': 'inativo',
                    'finalidades': [
                        {'nome': 'suporte', 'status': 'ativo'},
                        {'nome': 'marketing', 'status': 'inativo'}
                    ]
                },
                {
                    'id': 4,
                    'tipo': 'telefone',
                    'valor': '11988887777',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'suporte', 'status': 'ativo'},
                        {'nome': 'emergencia', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 5,
                    'tipo': 'email',
                    'valor': 'email3@gmail.com',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'vendas', 'status': 'ativo'}
                    ]
                }
            ]
        }
    },
    {
        'empresa': 'Metalúrgica S.A.',
        'cnpj': '98765432000101',
        'meioContato': {
            'contatos': [
                {
                    'id': 6,
                    'tipo': 'telefone',
                    'valor': '11922221111',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'geral', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 7,
                    'tipo': 'email',
                    'valor': 'email5@gmail.com',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'rh', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 8,
                    'tipo': 'telefone',
                    'valor': '11922223333',
                    'status': 'inativo',
                    'finalidades': [
                        {'nome': 'financeiro', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 9,
                    'tipo': 'email',
                    'valor': 'email6@gmail.com',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'compras', 'status': 'ativo'}
                    ]
                },
                {
                    'id': 10,
                    'tipo': 'telefone',
                    'valor': '11922224444',
                    'status': 'ativo',
                    'finalidades': [
                        {'nome': 'recepcao', 'status': 'ativo'}
                    ]
                }
            ]
        }
    }
]

# 3. Criar o DataFrame PySpark
# O PySpark pode inferir o schema, mas definir explicitamente é uma boa prática
# para dados complexos e para evitar erros de inferência.
schema = StructType([
    StructField("empresa", StringType(), True),
    StructField("cnpj", StringType(), True),
    StructField("meioContato", StructType([
        StructField("contatos", ArrayType(StructType([
            StructField("id", IntegerType(), True),
            StructField("tipo", StringType(), True),
            StructField("valor", StringType(), True),
            StructField("status", StringType(), True),
            StructField("finalidades", ArrayType(StructType([
                StructField("nome", StringType(), True),
                StructField("status", StringType(), True)
            ])), True)
        ])), True)
    ]), True)
])

df = spark.createDataFrame(data_raw, schema=schema)

print("--- DataFrame PySpark Original (aninhado) ---")
df.printSchema() # Mostra a estrutura do DataFrame
df.show(truncate=False) # Mostra o conteúdo


# 4. Processo para achatar o DataFrame

# ETAPA 1: Explodir a lista de contatos
# Acessa a lista 'contatos' dentro de 'meioContato' e usa explode
df_exploded_contacts = df.select(
    col("empresa"),
    col("cnpj"),
    explode(col("meioContato.contatos")).alias("contato") # 'contato' agora é uma StructType
)

print("\n--- DataFrame PySpark após explodir contatos ---")
df_exploded_contacts.printSchema()
df_exploded_contacts.show(truncate=False)

# ETAPA 2: Achatar os campos do 'contato' e preparar para finalidades
df_contacts_flat = df_exploded_contacts.select(
    col("empresa"),
    col("cnpj"),
    col("contato.id").alias("contato_id"),
    col("contato.tipo").alias("contato_tipo"),
    col("contato.valor").alias("contato_valor"),
    col("contato.status").alias("contato_status"),
    col("contato.finalidades").alias("finalidades_list") # Renomeia para clareza na próxima etapa
)

print("\n--- DataFrame PySpark com campos de contato achatados (finalidades ainda lista) ---")
df_contacts_flat.printSchema()
df_contacts_flat.show(truncate=False)


# ETAPA 3: Explodir e achatar a lista de finalidades
# Primeiro, explode a lista 'finalidades_list'
df_exploded_finalities = df_contacts_flat.select(
    col("empresa"),
    col("cnpj"),
    col("contato_id"),
    col("contato_tipo"),
    col("contato_valor"),
    col("contato_status"),
    explode(col("finalidades_list")).alias("finalidade") # 'finalidade' agora é uma StructType
)

# Agora, achata os campos de 'finalidade'
df_fully_flattened = df_exploded_finalities.select(
    col("empresa"),
    col("cnpj"),
    col("contato_id"),
    col("contato_tipo"),
    col("contato_valor"),
    col("contato_status"),
    col("finalidade.nome").alias("finalidade_nome"),
    col("finalidade.status").alias("finalidade_status")
)

print("\n--- DataFrame PySpark TOTALMENTE ACHATADO ---")
df_fully_flattened.printSchema()
df_fully_flattened.show(truncate=False)


# 5. Salvar o DataFrame final em um arquivo CSV
output_csv_path = "contatos_empresas_finalidades_pyspark.csv"
# Spark salva como diretório com múltiplos arquivos CSV (partições)
# Se quiser um único arquivo, pode repartitionar para 1 antes de escrever,
# mas isso pode afetar o desempenho em grandes datasets.
df_fully_flattened.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_csv_path)

print(f"\nDataFrame salvo com sucesso em '{output_csv_path}'")

# Para ver o conteúdo de um único arquivo CSV se você repartitionar para 1
# Exemplo (NÃO FAÇA ISSO PARA DATASETS GRANDES):
# df_fully_flattened.repartition(1).write \
#     .mode("overwrite") \
#     .option("header", "true") \
#     .csv("contatos_empresas_finalidades_pyspark_single_file.csv")

# 6. Parar a SparkSession
spark.stop()