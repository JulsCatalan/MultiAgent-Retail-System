import csv
import oracledb
from openai import OpenAI

client = OpenAI()

connection = oracledb.connect(
    user="ADMIN",
    password="tu_password",
    dsn="tu_tnsname_autonomousdb"
)
cursor = connection.cursor()

with open("products.csv", newline='', encoding="utf8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Generar embedding
        text = f"{row['prod_name']} {row['detail_desc']}"
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        # Insertar en Oracle
        cursor.execute("""
            INSERT INTO products (
                article_id, product_code, prod_name,
                product_type_name, product_group_name,
                graphical_appearance_name, colour_group_name,
                perceived_colour_value_name,
                perceived_colour_master_name,
                department_name, index_group_name,
                section_no, section_name, detail_desc,
                price_mxn, image_url, embedding
            )
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9,
                    :10, :11, :12, :13, :14, :15, :16, VECTOR(:17))
        """, (
            row["article_id"], row["product_code"], row["prod_name"],
            row["product_type_name"], row["product_group_name"],
            row["graphical_appearance_name"], row["colour_group_name"],
            row["perceived_colour_value_name"],
            row["perceived_colour_master_name"],
            row["department_name"], row["index_group_name"],
            row["section_no"], row["section_name"], row["detail_desc"],
            float(row["price_mxn"]), row["image_url"], emb
        ))

connection.commit()
