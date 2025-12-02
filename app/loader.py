# app/loader.py
import pandas as pd
import json
from .db import get_connection
from .embeddings import embed_text

def load_products_to_db(csv_path: str):
    df = pd.read_csv(csv_path)

    conn = get_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        # MEJORA: Texto natural optimizado para búsqueda semántica
        text_to_embed = generate_searchable_text(row)
        embedding = embed_text(text_to_embed)

        cur.execute("""
            INSERT OR REPLACE INTO products (
                article_id,
                product_code,
                prod_name,
                product_type_name,
                product_group_name,
                graphical_appearance_name,
                colour_group_name,
                perceived_colour_value_name,
                perceived_colour_master_name,
                department_name,
                index_group_name,
                section_no,
                section_name,
                detail_desc,
                price_mxn,
                image_url,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["article_id"],
            row["product_code"],
            row["prod_name"],
            row["product_type_name"],
            row["product_group_name"],
            row["graphical_appearance_name"],
            row["colour_group_name"],
            row["perceived_colour_value_name"],
            row["perceived_colour_master_name"],
            row["department_name"],
            row["index_group_name"],
            row["section_no"],
            row["section_name"],
            row["detail_desc"],
            row["price_mxn"],
            row["image_url"],
            json.dumps(embedding)
        ])

    conn.commit()
    conn.close()


def generate_searchable_text(row: pd.Series) -> str:
    """
    Genera texto optimizado para búsqueda semántica.
    
    Estrategia:
    1. Información más importante primero (nombre, tipo, categoría)
    2. Repetir términos clave para mayor peso semántico
    3. Texto natural sin etiquetas artificiales
    4. Incluir sinónimos y términos relacionados
    """
    parts = []
    
    # 1. NÚCLEO: Nombre del producto (máxima importancia - lo repetimos)
    prod_name = str(row.get('prod_name', '')).strip()
    if prod_name:
        parts.append(prod_name)
        parts.append(prod_name)  # Repetir para mayor peso
    
    # 2. TIPO Y CATEGORÍA (muy importante para búsquedas)
    product_type = str(row.get('product_type_name', '')).strip()
    category = str(row.get('product_group_name', '')).strip()
    
    if product_type and category:
        # Texto natural que incluye ambos
        parts.append(f"{product_type} de {category}")
    elif product_type:
        parts.append(product_type)
    elif category:
        parts.append(category)
    
    # 3. COLOR (muy buscado por usuarios)
    colour = str(row.get('colour_group_name', '')).strip()
    if colour and colour.lower() != 'nan':
        parts.append(f"color {colour}")
        parts.append(colour)  # Repetir para peso
    
    # 4. DEPARTAMENTO (contexto útil)
    department = str(row.get('department_name', '')).strip()
    if department and department.lower() != 'nan':
        parts.append(f"para {department}")
    
    # 5. ESTILO/APARIENCIA (diferenciador importante)
    style = str(row.get('graphical_appearance_name', '')).strip()
    if style and style.lower() not in ['nan', 'solid']:
        parts.append(f"estilo {style}")
    
    # 6. DESCRIPCIÓN DETALLADA (contexto rico)
    description = str(row.get('detail_desc', '')).strip()
    if description and description.lower() != 'nan':
        # Limpiar la descripción
        description = description.replace('\n', ' ').strip()
        if len(description) > 10:  # Solo si tiene contenido útil
            parts.append(description)
    
    # 7. TÉRMINOS DE BÚSQUEDA ADICIONALES (para mejorar recall)
    search_terms = []
    
    # Agregar sinónimos de categoría
    if category:
        category_lower = category.lower()
        if 'garment' in category_lower or 'ropa' in category_lower:
            search_terms.append("prenda de vestir")
        if 'accessories' in category_lower or 'accesorios' in category_lower:
            search_terms.append("complemento")
    
    # Agregar términos de departamento
    if department:
        dept_lower = department.lower()
        if 'women' in dept_lower or 'mujer' in dept_lower:
            search_terms.extend(["dama", "femenino"])
        elif 'men' in dept_lower or 'hombre' in dept_lower:
            search_terms.extend(["caballero", "masculino"])
        elif 'kids' in dept_lower or 'niño' in dept_lower:
            search_terms.extend(["infantil", "niños"])
    
    if search_terms:
        parts.append(" ".join(search_terms))
    
    # Unir todo con espacios (texto natural)
    final_text = " ".join(parts)
    
    # Limpieza final
    final_text = " ".join(final_text.split())  # Eliminar espacios múltiples
    
    return final_text