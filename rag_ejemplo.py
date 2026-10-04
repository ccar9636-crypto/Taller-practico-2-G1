"""
Taller Práctico #2 - Sistema RAG para EcoMarket
Utiliza LangChain, ChromaDB y Embeddings para responder consultas de atención al cliente.
"""

import os
import json
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings


def cargar_documentos() -> List[Document]:
    """Carga los 3 tipos de fuentes de datos de la carpeta docs/."""
    documentos = []
    
    # 1. Documento no estructurado (Política de devoluciones)
    if os.path.exists("docs/politica_devoluciones.txt"):
        loader = TextLoader("docs/politica_devoluciones.txt", encoding="utf-8")
        docs_politica = loader.load()
        for d in docs_politica:
            d.metadata["fuente"] = "Politica de Devoluciones"
        documentos.extend(docs_politica)

    # 2. Documento estructurado (Inventario JSON)
    if os.path.exists("docs/inventario_productos.json"):
        with open("docs/inventario_productos.json", "r", encoding="utf-8") as f:
            inventario = json.load(f)
            for item in inventario:
                contenido = f"Producto: {item['nombre']}. Categoria: {item['categoria']}. Precio: ${item['precio']} USD. Stock: {item['stock']} unidades. Descripcion: {item['descripcion']}"
                documentos.append(Document(page_content=contenido, metadata={"fuente": "Inventario", "id": item["id"]}))

    # 3. Documento semi-estructurado (Preguntas Frecuentes JSON)
    if os.path.exists("docs/preguntas_frecuentes.json"):
        with open("docs/preguntas_frecuentes.json", "r", encoding="utf-8") as f:
            faqs = json.load(f)
            for faq in faqs:
                contenido = f"Pregunta Frecuente: {faq['pregunta']} Respuesta: {faq['respuesta']}"
                documentos.append(Document(page_content=contenido, metadata={"fuente": "Preguntas Frecuentes"}))

    return documentos


def crear_sistema_rag():
    print("[1/3] Cargando documentos de EcoMarket...")
    documentos = cargar_documentos()
    print(f"   Total de documentos base cargados: {len(documentos)}")

    print("\n[2/3] Aplicando Chunking (RecursiveCharacterTextSplitter)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"   Total de fragmentos (chunks) generados: {len(chunks)}")

    print("\n[3/3] Creando vectores e indexando en ChromaDB...")
    embeddings = FastEmbedEmbeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ecomarket_knowledge"
    )
    print("   Base de datos vectorial indexada exitosamente.")

    return vectorstore


def consultar_rag(vectorstore, pregunta: str, k: int = 2):
    print(f"\n[CONSULTA] Consulta del cliente: '{pregunta}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    resultados = retriever.invoke(pregunta)
    
    print("\n[RESULTADOS] Fragmentos recuperados por similitud semántica (RAG):")
    for i, doc in enumerate(resultados, 1):
        fuente = doc.metadata.get("fuente", "Desconocida")
        print(f"\n--- Resultado #{i} (Fuente: {fuente}) ---")
        print(doc.page_content)


if __name__ == "__main__":
    # Inicializar el sistema RAG
    vectorstore = crear_sistema_rag()
    
    # Pruebas con preguntas reales de clientes de EcoMarket
    consultar_rag(vectorstore, "¿Cuántos días tengo para devolver un producto defectuoso?")
    consultar_rag(vectorstore, "¿Tienen cepillos de bambú y cuánto cuestan?")
    consultar_rag(vectorstore, "¿Los empaques son ecológicos?")
