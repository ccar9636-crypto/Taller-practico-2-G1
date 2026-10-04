# Taller Práctico #2: Sistema RAG para EcoMarket
**Caso de Estudio:** Optimización de la Atención al Cliente en E-commerce mediante Generación Aumentada por Recuperación (RAG).

**Integrantes:**
- Deibi Bastidas Cerón.
- Camilo Arciniegas Forero.
---

## Fase 1: Selección de Componentes Clave del Sistema RAG

### 1. Modelo de Embeddings
**Selección:** `text-embedding-3-small` (OpenAI) / Alternativa Open-Source: `intfloat/multilingual-e5-base` (HuggingFace) o `FastEmbed`.

**Justificación:**
* **Precisión y Multilingüismo:** Ambos modelos destacan en la captura de matices semánticos en español, fundamental para la atención al cliente de EcoMarket.
* **Costo y Escalabilidad:** `text-embedding-3-small` de OpenAI es extremadamente económico ($0.02 por cada millón de tokens) y ofrece dimensiones ajustables (1536 dimensiones por defecto), reduciendo almacenamiento sin perder precisión.
* **Propietario vs. Open Source:** 
  * *OpenAI (`text-embedding-3-small`):* Modelo propietario que no requiere infraestructura dedicada ni mantenimiento de GPUs.
  * *HuggingFace / FastEmbed (`multilingual-e5-base`):* Modelos de código abierto ideales si EcoMarket requiere privacidad total y ejecución on-premise sin costo de suscripción por token.

---

### 2. Base de Datos Vectorial
**Selección:** `ChromaDB` (Entorno de Desarrollo/Prototipado) con visión a `Qdrant` / `Pinecone` (Producción).

**Justificación y Comparativa:**
* **ChromaDB:** Base de datos vectorial de código abierto, embebida y liviana. Se integra nativamente con LangChain/LlamaIndex sin costo de infraestructura, ideal para el prototipo de EcoMarket.
* **Pinecone:** Servicio SaaS completamente administrado en la nube. Excelente escalabilidad pero genera costo continuo y dependencia de terceros.
* **Weaviate / Qdrant:** Ofrecen búsqueda híbrida (vectorial + palabras clave BM25), altamente recomendadas si EcoMarket crece a miles de productos.

---

## Fase 2: Creación de la Base de Conocimiento de Documentos

### 1. Documentos Seleccionados (Base de Conocimiento)
Para atender las solicitudes de los clientes de EcoMarket, se identificaron 3 tipos de fuentes clave en la carpeta `docs/`:
1. **`politica_devoluciones.txt` (Documento No Estructurado):** Contiene reglas de garantías, devoluciones, plazos y condiciones de envíos ecológicos.
2. **`inventario_productos.json` (Documento Estructurado):** Catálogo con IDs, nombres de productos ecológicos, stock, precios y categorías.
3. **`preguntas_frecuentes.json` (Documento Semi-estructurado):** Pares de Pregunta-Respuesta sobre sostenibilidad, empaques biodegradables y métodos de pago.

---

### 2. Estrategia de Segmentación (Chunking)
* **Estrategia Elegida:** *RecursiveCharacterTextSplitter* (Segmentación por caracteres recursiva).
* **Parámetros:** `chunk_size=300` caracteres, `chunk_overlap=50` caracteres.
* **Justificación:**
  * La división recursiva respeta la estructura natural del lenguaje (párrafos `\n\n`, frases `\n`, espacios `" "`), evitando cortar frases a la mitad.
  * El solapamiento (*overlap*) de 50 caracteres preserva el contexto entre fragmentos contiguos, evitando que información relevante en los límites quede aislada.

---

### 3. Proceso de Indexación
1. **Carga (Loading):** Se leen los archivos de la carpeta `docs/`.
2. **Fragmentación (Splitting):** Se aplica el *Chunking* recursivo.
3. **Vectorización (Embedding):** Cada chunk se pasa por el modelo de embeddings para generar un vector representativo.
4. **Almacenamiento (Indexing):** Los vectores y sus metadatos (origen, categoría) se insertan en ChromaDB.

---

## Fase 3: Integración y Ejecución de Código

El script ejecutable se encuentra en `rag_ejemplo.py`.

### Limitaciones y Suposiciones:
1. **Recursos de Cómputo:** Se utiliza un modelo de embeddings ligero local para garantizar la ejecución fluida en entornos locales sin depender de claves API de pago.
2. **Escalabilidad Vectorial:** ChromaDB en modo embebido es óptimo para prototipos; para un entorno de producción con millones de registros se asumiría la migración a un clúster de Qdrant o Pinecone.

---

## Instrucciones de Instalación y Ejecución

Este proyecto utiliza **`uv`** como gestor moderno de entornos virtuales y dependencias para Python.

### Requisitos Previos
* Python 3.10 o superior
* `uv` instalado (`brew install uv` o `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Pasos de Ejecución

1. **Sincronizar el entorno virtual e instalar dependencias:**
   ```bash
   uv sync
   ```

2. **Ejecutar el sistema RAG de atención al cliente:**
   ```bash
   uv run rag_ejemplo.py
   ```

3. **Ejecutar la evaluación cuantitativa de métricas y generación de gráficos:**
   ```bash
   uv run evaluacion_metricas.py
   ```
   *Nota: Este comando genera automáticamente el reporte gráfico `metricas_rag.png`.*