# rag_demo_setup.py
import asyncio
from config import ConfigManager
from rag import create_enhanced_rag_pipeline

async def main():
    cfg_manager = ConfigManager()
    cfg = cfg_manager.get_config()

    rag = create_enhanced_rag_pipeline(cfg)

    if not rag.enabled:
        print("❌ KnowledgeBase 在配置中是关闭的（knowledge_base.enabled=false）")
        return

    print("✅ RAG 已启用，开始索引 demo 文档...")

    texts = [
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，用来让大模型在回答问题时参考外部知识库。",
        "分散投资可以通过持有多种资产来降低组合风险，比如同时持有股票、债券和现金，从而减少单一资产波动对整体收益的影响。",
    ]
    metadatas = [
        {"source": "demo", "topic": "RAG"},
        {"source": "demo", "topic": "投资-分散化"},
    ]

    # 这里不传 user_id，走“通用知识库”
    rag.index_documents(
        texts=texts,
        metadatas=metadatas,
        user_id=None,
    )

    print("🎉 demo 文档索引完成！现在可以用 agent.py 来问问题了。")

if __name__ == "__main__":
    asyncio.run(main())